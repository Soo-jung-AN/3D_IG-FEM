"""Vp from the volumetric strain of a DEM run.

The premise, in one line: a discrete-element model gives you particle
displacements; those give you a finite strain; and the volumetric part of
that strain tells you how the rock's velocity changed. That is the whole
claim of the Botter et al. (2014) workflow, and this script is the
smallest honest implementation of it.

  1. deformation gradient F at every particle, by the IG-FEM projection
     of An et al. (2023) over a Delaunay tetrahedralisation
  2. volumetric strain  ev = det(F) - 1
  3. reference Vp from the model's OWN per-particle density, through the
     Nafe-Drake curve (Brocher, 2005) -- no free parameter
  4. Vp = Vp_ini * f(ev), Botter et al. Eq. (3): compaction speeds the
     rock up, dilatation slows it down, bounded to +/-25%
  5. density by mass conservation, rho = rho_0 / det(F)
  6. Vs from Brocher's regression fit, which stays physical down to
     1.5 km/s (Han's sandstone line does not)

Usage:
  python3 vp_from_strain.py --init init_pos.txt --pos pos.txt \
                            --density density.txt [--out out.npz]
"""
import argparse
import time

import numpy as np
from scipy.spatial import Delaunay, cKDTree

import rock_physics as R
from preprocessing3 import tet_quality
from run_stage import vol_strain

Q_MIN = 0.05
ALPHA_FACTOR = 3.0     # max tetrahedron edge, in median nearest-neighbour spacings


def orient_positive(X0, tets):
    """Reorder each tetrahedron so its signed volume is positive.

    scipy's Delaunay returns simplices in arbitrary orientation -- 50% of
    them negative for this pack -- while the assembly in Assembly3.py
    weights every element integral by det(J) without an absolute value.
    Negative-volume elements therefore SUBTRACT mass from the global
    matrix, which makes it indefinite and the recovered deformation
    gradient meaningless (det(F) - 1 reaching 1e10 here). The ParaView
    mesh in txt/ happens to be consistently oriented, which is why
    main.py never had to do this. Swapping the last two vertices flips
    the sign."""
    a, b, c, d = (X0[tets[:, k]] for k in range(4))
    neg = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) < 0
    tets = tets.copy()
    tets[neg, 2], tets[neg, 3] = tets[neg, 3], tets[neg, 2].copy()
    return tets, int(neg.sum())


def max_edge(X0, tets):
    a, b, c, d = (X0[tets[:, k]] for k in range(4))
    return np.stack([np.linalg.norm(x - y, axis=1) for x, y in
                     ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))], axis=1).max(axis=1)


def build_mesh(X0, q_min=Q_MIN, alpha=None, verbose=True):
    """Delaunay tetrahedralisation of the undeformed positions, filtered
    twice.

    A Delaunay triangulation fills the CONVEX HULL, so a particle pack
    with a free surface and concave fault-bounded geometry gets long thin
    elements bridging across the empty space -- here up to a 1985 m edge
    in a 2 km model whose particles sit 41 m apart. Those elements
    average the displacement across the whole domain and destroy the
    recovered deformation gradient (median det(F) - 1 of -705 without the
    filter, against -0.01 for an independent nearest-neighbour estimate).
    The repository's 2D code has always applied this alpha filter; it is
    not needed for the 3D model in txt/ only because that mesh was built
    in ParaView with the criterion already applied.

    So: drop elements with any edge longer than alpha (default 3x the
    median nearest-neighbour spacing), then drop the remaining slivers on
    shape quality."""
    tets = Delaunay(X0).simplices.astype(np.int64)
    tets, n_flipped = orient_positive(X0, tets)
    if verbose and n_flipped:
        print(f"  {n_flipped} of {len(tets)} tetrahedra re-oriented to positive volume")
    if alpha is None:
        d, _ = cKDTree(X0).query(X0, k=2)
        alpha = ALPHA_FACTOR * np.median(d[:, 1])
    long_edge = max_edge(X0, tets) > alpha
    sliver = tet_quality(X0, tets) < q_min
    keep = ~(long_edge | sliver)
    if verbose:
        print(f"  {len(tets)} tetrahedra: {100*long_edge.mean():.2f}% over alpha = "
              f"{alpha:.0f} m, {100*sliver.mean():.2f}% slivers (q < {q_min}), "
              f"{100*keep.mean():.2f}% kept")
    orphan = np.setdiff1d(np.arange(len(X0)), np.unique(tets[keep]))
    if len(orphan) and verbose:
        print(f"  WARNING: {len(orphan)} particles have no supporting element")
    return tets[keep], orphan


def sspx_vol_strain(X0, X1, k=16):
    """Nearest-neighbour least-squares deformation gradient, for a check."""
    _, idx = cKDTree(X0).query(X0, k=k + 1)
    idx = idx[:, 1:]
    dX = X0[idx] - X0[:, None, :]
    dx = X1[idx] - X1[:, None, :]
    A = np.einsum("nkj,nkl->njl", dX, dX)
    B = np.einsum("nkj,nkl->njl", dx, dX)
    return np.linalg.det(np.matmul(B, np.linalg.inv(A))) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--density", required=True)
    ap.add_argument("--out", default="./results/vp_from_strain.npz")
    ap.add_argument("--q-min", type=float, default=Q_MIN)
    ap.add_argument("--alpha", type=float, default=None,
                    help="max tetrahedron edge in m (default 3x the median "
                         "nearest-neighbour spacing)")
    args = ap.parse_args()
    t0 = time.time()

    X0 = np.loadtxt(args.init)
    X1 = np.loadtxt(args.pos)
    rho_0 = np.loadtxt(args.density)
    print(f"{len(X0)} particles, "
          f"{(X0[:,0].max()-X0[:,0].min())/1e3:.2f} x "
          f"{(X0[:,1].max()-X0[:,1].min())/1e3:.2f} x "
          f"{(X0[:,2].max()-X0[:,2].min())/1e3:.2f} km", flush=True)

    ele_id, orphan = build_mesh(X0, args.q_min, args.alpha)
    vol = vol_strain(X0, X1, ele_id)
    vol_s = sspx_vol_strain(X0, X1)
    both = (np.abs(vol) < 1) & (np.abs(vol_s) < 1)
    print(f"\nvolumetric strain  IG-FEM  mean {vol.mean():+.4f}  median {np.median(vol):+.4f}"
          f"  p1 {np.percentile(vol,1):+.3f}  p99 {np.percentile(vol,99):+.3f}")
    print(f"                   SSPX     mean {vol_s.mean():+.4f}  median {np.median(vol_s):+.4f}"
          f"   corr {np.corrcoef(vol[both], vol_s[both])[0,1]:+.3f} (n={both.sum()})")

    # reference velocity from the model's own densities
    Vp_ini = R.vp_from_density_nafe_drake(rho_0)          # km/s
    Vp = R.vp_from_strain(vol, Vp_ini)                     # km/s
    Vs = R.vs_from_vp_brocher(Vp)                          # km/s
    rho = R.density_from_mass_conservation(rho_0, vol)
    Z = rho * Vp * 1000.0

    print("\nreference state, by density layer:")
    print(f"  {'rho':>6} {'n':>7} {'depth m':>17} {'Vp_ini':>8} {'Vs_ini':>8} {'Vp/Vs':>7}")
    for v in np.unique(rho_0):
        m = rho_0 == v
        vp0 = Vp_ini[m][0]
        print(f"  {v:6.0f} {m.sum():7d} {X0[m,2].min():8.0f}..{X0[m,2].max():7.0f}"
              f" {vp0:8.3f} {R.vs_from_vp_brocher(vp0):8.3f} {vp0/R.vs_from_vp_brocher(vp0):7.3f}")

    print("\nafter deformation:")
    for name, a, u in (("Vp", Vp, "km/s"), ("Vs", Vs, "km/s"), ("Vp/Vs", Vp / Vs, ""),
                       ("rho", rho, "kg/m3"), ("Z", Z / 1e6, "e6")):
        print(f"  {name:>5} {u:>6}   min {a.min():8.3f}   mean {a.mean():8.3f}"
              f"   median {np.median(a):8.3f}   max {a.max():8.3f}")
    print(f"\n  Vp change from the reference: mean {100*(Vp/Vp_ini-1).mean():+.2f}%  "
          f"p1 {100*np.percentile(Vp/Vp_ini-1,1):+.1f}%  p99 {100*np.percentile(Vp/Vp_ini-1,99):+.1f}%")
    print(f"  slowed by dilatation: {100*np.mean(vol>0):.1f}% of particles   "
          f"sped up by compaction: {100*np.mean(vol<0):.1f}%")

    np.savez_compressed(args.out, X0=X0, X1=X1, rho_0=rho_0, vol=vol, vol_sspx=vol_s,
                        Vp_ini=Vp_ini, Vp=Vp, Vs=Vs, rho=rho, Z=Z)
    print(f"\nsaved {args.out}   ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
