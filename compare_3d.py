"""Three-way comparison of synthetic seismic properties for the SAME 3D
DEM deformation (init_pos.txt -> m4_1_pos.txt, 259,943 particles).

  1. SSPX + Botter     nearest-neighbour local least-squares deformation
                       gradient (Cardozo & Allmendinger, 2009 -- the
                       strain method Botter et al. themselves used)
                       -> rock_physics.py Eqs. (1)-(4).
  2. IG-FEM + Botter   this repo's global mass-matrix L2 projection of
                       the element-wise deformation gradient (main.py)
                       -> the SAME Eqs. (1)-(4).
  3. Hertz-Mindlin     contact mechanics from first principles
                       (hertz_mindlin_3d.py). No strain, no empirical
                       curve.

Routes 1 and 2 share one reference state exactly -- rock_physics.ZONES,
the same phi_ini / rho_grain / Vp_ini arrays main.py uses -- so any
difference between them is a difference in the recovered strain and
nothing else. That is the point of the exercise: the 2D version of this
comparison found the two strain fields agree only weakly pointwise
(r = +0.06) yet give traveltimes within 0.14 s of each other, and it is
worth knowing whether that survives in 3D.

Each Vp field is interpolated onto a common regular 3D grid over the
deformed geometry, and the eikonal equation is solved in 3D with the
fast marching method from a single surface source, giving first-arrival
traveltimes. A synthetic seismic section (Botter et al. step 3 in its
lightweight 1D-convolution form) is then made on the mid-model y slice.

Usage:  python3 compare_3d.py [--dx 250] [--igfem-vol vol.npy]
"""
import argparse
import time

import numpy as np
import skfmm
from scipy.spatial import cKDTree

from rock_physics import synthesize_vpvs, zoned_initial_properties
import hertz_mindlin_3d

UNDEFORMED = "./txt/init_pos.txt"
DEFORMED = "./txt/m4_1_pos.txt"
RADIUS = "./txt/init_rad.txt"
VTK_RESULT = "./results/80-3.vtk"
OUT = "./results/compare_3d.npz"

DX = 250.0          # m, grid spacing
KNN = 8             # neighbours used to interpolate particles -> grid
INSIDE_FACTOR = 2.0 # a grid node is "inside" if a particle is within this*dx
SEISMIC_FREQ = 30.0 # Hz, dominant frequency of the Ricker wavelet
DT = 0.002          # s, time sampling of the convolution


# ----------------------------------------------------------------- strain

def sspx_vol_strain(X0, X1, k=16, chunk=20000):
    """Volumetric strain from a nearest-neighbour local least-squares fit
    of F to dx = F dX (the SSPX approach of Cardozo & Allmendinger, 2009).
    Chunked: 260k x 16 x 3 x 3 at once is avoidable."""
    tree = cKDTree(X0)
    out = np.empty(len(X0))
    for s in range(0, len(X0), chunk):
        sl = slice(s, min(s + chunk, len(X0)))
        _, idx = tree.query(X0[sl], k=k + 1)
        idx = idx[:, 1:]                                  # drop self
        dX = X0[idx] - X0[sl, None, :]
        dx = X1[idx] - X1[sl, None, :]
        A = np.einsum("nkj,nkl->njl", dX, dX)
        B = np.einsum("nkj,nkl->njl", dx, dX)
        out[sl] = np.linalg.det(np.linalg.solve(
            A.transpose(0, 2, 1), B.transpose(0, 2, 1)).transpose(0, 2, 1)) - 1.0
    return out


def igfem_vol_from_vtk(path, name="vol"):
    """Read one POINT_DATA scalar out of the .vtk main.py writes, so this
    script uses the IG-FEM strain exactly as the solver produced it rather
    than re-running the 16-minute solve."""
    values = []
    with open(path) as f:
        for line in f:
            if line.startswith("SCALARS ") and line.split()[1] == name:
                next(f)                                   # LOOKUP_TABLE
                for line in f:
                    if line.startswith("SCALARS "):
                        break
                    values.extend(line.split())
                break
    if not values:
        raise ValueError(f"scalar {name!r} not found in {path}")
    return np.array(values, dtype=np.float64)


# ------------------------------------------------------------ gridding

def make_grid(pos, dx):
    axes = [np.arange(pos[:, d].min(), pos[:, d].max() + dx, dx) for d in range(3)]
    shape = tuple(len(a) for a in axes)
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return axes, shape, nodes


def interpolate(tree, values, nodes, shape, k=KNN, chunk=500000):
    """Inverse-distance-weighted average of the k nearest particles. The
    grid is finer than the particle spacing, so plain nearest-neighbour
    would be visibly blocky; a Delaunay-based linear interpolant over
    260k points in 3D is not worth its cost here."""
    out = np.empty(len(nodes))
    for s in range(0, len(nodes), chunk):
        sl = slice(s, min(s + chunk, len(nodes)))
        d, i = tree.query(nodes[sl], k=k)
        w = 1.0 / np.maximum(d, 1e-9) ** 2
        out[sl] = (w * values[i]).sum(axis=1) / w.sum(axis=1)
    return out.reshape(shape)


# ------------------------------------------------------ synthetic seismic

def ricker(f, dt, length=0.512):
    t = np.arange(-length / 2, length / 2 + dt, dt)
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def trace_synthetic(dz, vp, rho, f, dt=DT):
    """One zero-offset trace: depth-sampled Vp/rho (downward, uniform dz)
    -> impedance -> two-way time -> normal-incidence reflectivity ->
    Ricker convolution -> back to depth."""
    if len(vp) < 4:
        return np.zeros_like(vp)
    twt = np.concatenate([[0.0], np.cumsum(2 * dz / vp[:-1])])
    nt = int(np.ceil(twt[-1] / dt)) + 1
    t_uniform = np.arange(nt) * dt
    Z_t = np.interp(t_uniform, twt, rho * vp)
    refl = np.zeros(nt)
    refl[:-1] = np.diff(Z_t) / (Z_t[:-1] + Z_t[1:])
    return np.interp(twt, t_uniform, np.convolve(refl, ricker(f, dt), mode="same"))


def section(vp_slice, rho_slice, inside_slice, dz, f):
    """vp/rho slices are (nx, nz) with z increasing upward."""
    nx, nz = vp_slice.shape
    out = np.full((nx, nz), np.nan)
    for ix in range(nx):
        col = np.where(inside_slice[ix])[0]
        if len(col) < 4:
            continue
        rows = np.arange(col.max(), col.min() - 1, -1)     # downward
        out[ix, rows] = trace_synthetic(dz, vp_slice[ix, rows], rho_slice[ix, rows], f)
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--igfem-vol", default=None,
                    help="npy with the IG-FEM volumetric strain "
                         "(default: read it back out of results/80-3.vtk)")
    args = ap.parse_args()
    t0 = time.time()

    X0 = np.loadtxt(UNDEFORMED)
    X1 = np.loadtxt(DEFORMED)
    rad = np.loadtxt(RADIUS)
    print(f"{len(X0)} particles")

    vol_igfem = (np.load(args.igfem_vol) if args.igfem_vol
                 else igfem_vol_from_vtk(VTK_RESULT))
    print("SSPX strain ...", flush=True)
    vol_sspx = sspx_vol_strain(X0, X1)

    both = (np.abs(vol_sspx) < 1) & (np.abs(vol_igfem) < 1)
    print(f"volumetric strain   SSPX mean {vol_sspx.mean():+.4f} median {np.median(vol_sspx):+.4f}"
          f"   IG-FEM mean {vol_igfem.mean():+.4f} median {np.median(vol_igfem):+.4f}")
    print(f"                    corr: all {np.corrcoef(vol_sspx, vol_igfem)[0, 1]:+.3f}"
          f"   |vol|<1 both (n={both.sum()}) {np.corrcoef(vol_sspx[both], vol_igfem[both])[0, 1]:+.3f}")

    phi_ini, rho_g, Vp_ini = zoned_initial_properties(X0)

    def botter(vol):
        _, rho, Vp_kms, _, _ = synthesize_vpvs(vol, phi_ini, rho_g, Vp_ini)
        return Vp_kms * 1000.0, rho

    print("Hertz-Mindlin ...", flush=True)
    hm = hertz_mindlin_3d.run(X1, rad)

    vp_s, rho_s = botter(vol_sspx)
    vp_i, rho_i = botter(vol_igfem)
    names = ["SSPX + Botter", "IG-FEM + Botter", "Hertz-Mindlin"]
    fields = {
        "SSPX + Botter":   (vp_s, rho_s, None),
        "IG-FEM + Botter": (vp_i, rho_i, None),
        "Hertz-Mindlin":   (hm["Vp"], hm["rho"], hm["valid"]),
    }
    print()
    for n in names:
        vp, rho, m = fields[n]
        sel = np.isfinite(vp) if m is None else m
        print(f"{n:>16s}  Vp (m/s) min {vp[sel].min():6.0f}  mean {vp[sel].mean():6.0f}  "
              f"max {vp[sel].max():6.0f}   rho {rho[sel].mean():5.0f} kg/m3   n={sel.sum()}")

    axes, shape, nodes = make_grid(X1, args.dx)
    gx, gy, gz = axes
    print(f"\ngrid {shape[0]} x {shape[1]} x {shape[2]} = {np.prod(shape)/1e6:.2f}M nodes, "
          f"dx = {args.dx:.0f} m", flush=True)

    tree_all = cKDTree(X1)
    dist, _ = tree_all.query(nodes)
    inside = (dist < INSIDE_FACTOR * args.dx).reshape(shape)
    print(f"inside the pack: {100 * inside.mean():.1f}% of nodes")

    # source at the top of the model, mid-way in x and y
    src = [np.argmin(np.abs(gx - 0.5 * (gx[0] + gx[-1]))),
           np.argmin(np.abs(gy - 0.5 * (gy[0] + gy[-1]))), 0]
    src[2] = np.where(inside[src[0], src[1]])[0].max()
    print(f"source at x={gx[src[0]]/1000:.1f} km, y={gy[src[1]]/1000:.1f} km, "
          f"z={gz[src[2]]/1000:.2f} km")
    phi_lsf = np.ones(shape)
    phi_lsf[tuple(src)] = -1.0

    jy = src[1]                                  # slice for the seismic section
    vp_grids, rho_grids, tts, secs = [], [], [], []
    for n in names:
        vp, rho, m = fields[n]
        tree = tree_all if m is None else cKDTree(X1[m])
        vpv, rhov = (vp, rho) if m is None else (vp[m], rho[m])
        VP = interpolate(tree, vpv, nodes, shape)
        RHO = interpolate(tree, rhov, nodes, shape)
        T = skfmm.travel_time(phi_lsf, VP, dx=args.dx)
        tmax = np.ma.array(T, mask=~inside).max()
        S = section(VP[:, jy, :], RHO[:, jy, :], inside[:, jy, :], args.dx, SEISMIC_FREQ)
        print(f"{n:>16s}  gridded Vp mean {VP[inside].mean():6.0f} m/s   "
              f"max traveltime {tmax:.3f} s", flush=True)
        vp_grids.append(VP); rho_grids.append(RHO); tts.append(T); secs.append(S)

    print()
    ref = names.index("IG-FEM + Botter")
    m3 = inside
    for r, n in enumerate(names):
        dt_rms = np.sqrt(np.mean((tts[r][m3] - tts[ref][m3]) ** 2))
        cc = np.corrcoef(tts[r][m3], tts[ref][m3])[0, 1]
        ms = np.isfinite(secs[r]) & np.isfinite(secs[ref])
        cs = np.corrcoef(secs[r][ms], secs[ref][ms])[0, 1]
        print(f"{n:>16s} vs IG-FEM + Botter:  traveltime corr {cc:+.4f}  "
              f"RMS dt {dt_rms:.4f} s   |   section corr {cs:+.4f}")

    np.savez_compressed(
        OUT, gx=gx, gy=gy, gz=gz, dx=args.dx, inside=inside, slice_y=jy,
        src=np.array([gx[src[0]], gy[src[1]], gz[src[2]]]),
        names=np.array(names), vp_grids=np.array(vp_grids),
        rho_grids=np.array(rho_grids), tts=np.array(tts), secs=np.array(secs),
        vol_sspx=vol_sspx, vol_igfem=vol_igfem,
        hm_Vp=hm["Vp"], hm_valid=hm["valid"], hm_phi=hm["phi"], hm_P=hm["P"],
        hm_C=hm["C"], X1=X1)
    print(f"\nsaved {OUT}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
