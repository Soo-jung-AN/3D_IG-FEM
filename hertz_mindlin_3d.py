"""Hertz-Mindlin contact-based Vp/Vs for the 3D DEM model.

A first-principles alternative to the Botter et al. (2014) empirical
route in rock_physics.py: instead of mapping a finite volumetric strain
through fitted curves, the elastic moduli are built from the contact
mechanics of the pack itself.

    contacts from geometry -> Hertzian normal force from the overlap
    -> coordination number C, local porosity phi, local pressure P
    -> K_HM, G_HM (Mindlin 1949; Digby 1981; Walton 1987; see Mavko et
       al., The Rock Physics Handbook) -> Vp, Vs

This is the port of the 2D implementation in Soo-jung-AN/HM_DEM, and it
is the formulas' native setting: the Hertz-Mindlin effective-medium
expressions are derived for random packs of SPHERES, so the 2D
application was the approximation, not this one.

Two things differ from the 2D code, both forced by the size of this
model (259,943 particles, ~450 neighbours inside a measurement sphere,
i.e. 1.2e8 neighbour indices):

  * every neighbourhood sum is chunked rather than materialised at once;
  * the virial stress is coarse-grained properly. The 2D version divides
    ONE particle's own contacts by the whole measurement circle, which
    under-normalises the pressure by roughly the number of particles in
    the circle. Here each contact's f (x) l is split between its two
    particles first, so summing over the sphere reproduces the sum over
    every contact inside it -- which is what the sphere volume in the
    denominator is the volume of. Since Vp ~ P^(1/6) the 2D error is a
    near-uniform scale factor on Vp, but it is still an error.

Note on this DEM's contact convention: the pack is "soft", with median
centre-to-centre spacing 640 m against a median radius sum of 735 m, so
the geometric overlaps are ~13% of the grain size. Reconstructing
Hertzian forces from those overlaps gives a coarse-grained pressure of
~2.8 GPa, well above lithostatic at 15 km. The velocity level therefore
carries the DEM's stiffness convention, not an independently calibrated
in-situ stress; the informative part of this route is the spatial
pattern of Vp and its contrast with the strain-driven routes.
"""
import numpy as np
from scipy.spatial import cKDTree

# Quartz (typical sand/sandstone grain mineral), as in HM_DEM
E_GRAIN, NU_GRAIN, RHO_GRAIN = 94.5e9, 0.17, 2650.0

R_M_FACTOR = 8.0            # measurement-sphere radius, in mean particle radii
PHI_CLIP = (0.02, 0.60)     # plausible random-packing porosity range
CHUNK = 4000                # particles per neighbourhood-query block


def find_contacts(pos, rad, tol=1e-6):
    """Geometric contacts: pairs whose spheres overlap. Returns the pair
    indices, the overlap, and the unit branch direction."""
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=2 * rad.max(), output_type="ndarray")
    d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    dist = np.linalg.norm(d, axis=1)
    overlap = rad[pairs[:, 0]] + rad[pairs[:, 1]] - dist
    keep = overlap > tol
    return pairs[keep], overlap[keep], d[keep] / dist[keep, None]


def neighbour_sum(tree, pos, R_m, per_particle):
    """Sum a per-particle quantity, shape (n,) or (n, k), over every
    particle inside a measurement sphere of radius R_m. Chunked."""
    n = len(pos)
    flat = per_particle.reshape(n, -1)
    out = np.zeros((n, flat.shape[1]))
    for s in range(0, n, CHUNK):
        lists = tree.query_ball_point(pos[s:s + CHUNK], r=R_m)
        counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        idx = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        block = np.add.reduceat(flat[idx], starts, axis=0)
        block[counts == 0] = 0.0
        out[s:s + len(lists)] = block
    return out.reshape(per_particle.shape)


def hertz_mindlin_KG(C, phi, G_grain, nu, P):
    """Dry-pack effective bulk and shear moduli of a random sphere pack at
    confining pressure P. Note K/G = 5(2-nu) / (3(5-4nu)) independently of
    C, phi and P: this model pins Vp/Vs to a single number."""
    common = (C ** 2 * (1 - phi) ** 2 * G_grain ** 2) / (np.pi ** 2 * (1 - nu) ** 2)
    P = np.clip(P, 0.0, None)
    K = (common * P / 18.0) ** (1 / 3)
    G = ((5 - 4 * nu) / (5 * (2 - nu))) * (3 * common * P / 2.0) ** (1 / 3)
    return K, G


def run(pos, rad, e_grain=E_GRAIN, nu=NU_GRAIN, rho_grain=RHO_GRAIN, verbose=True):
    """Vp/Vs on the deformed pack. Velocities are in m/s, and are NaN
    outside `valid` (particles whose measurement sphere would stick out of
    the sample, or that carry no contact / no positive pressure)."""
    n = len(pos)
    pairs, overlap, normal = find_contacts(pos, rad)
    if verbose:
        print(f"  contacts: {len(pairs)}", flush=True)

    R_eff = (rad[pairs[:, 0]] * rad[pairs[:, 1]]) / (rad[pairs[:, 0]] + rad[pairs[:, 1]])
    E_star = e_grain / (2 * (1 - nu ** 2))
    Fn = (4.0 / 3.0) * E_star * np.sqrt(R_eff) * overlap ** 1.5

    C = np.zeros(n, dtype=np.int64)
    np.add.at(C, pairs[:, 0], 1)
    np.add.at(C, pairs[:, 1], 1)

    R_m = R_M_FACTOR * rad.mean()
    tree = cKDTree(pos)
    sphere_vol = 4.0 / 3.0 * np.pi * R_m ** 3

    # half of each contact's virial contribution to each of its particles,
    # so that a neighbourhood sum is the coarse-grained stress of the sphere
    branch = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    contrib = 0.5 * (Fn[:, None] * normal)[:, :, None] * branch[:, None, :]
    s_part = np.zeros((n, 3, 3))
    np.add.at(s_part, pairs[:, 0], contrib)
    np.add.at(s_part, pairs[:, 1], contrib)

    pvol = 4.0 / 3.0 * np.pi * rad ** 3
    summed = neighbour_sum(tree, pos, R_m, np.column_stack(
        [pvol, s_part[:, 0, 0], s_part[:, 1, 1], s_part[:, 2, 2]]))
    phi = 1.0 - summed[:, 0] / sphere_vol
    P = summed[:, 1:].sum(axis=1) / (3.0 * sphere_vol)
    if verbose:
        print(f"  R_m = {R_m:.0f} m, raw phi: min {phi.min():.3f} "
              f"median {np.median(phi):.3f} max {phi.max():.3f}", flush=True)

    lo, hi = pos.min(axis=0) + R_m, pos.max(axis=0) - R_m
    near_boundary = np.any((pos < lo) | (pos > hi), axis=1)
    phi = np.clip(phi, *PHI_CLIP)

    valid = ~near_boundary & (C > 0) & (P > 0) & np.isfinite(phi)
    K = np.full(n, np.nan)
    G = np.full(n, np.nan)
    K[valid], G[valid] = hertz_mindlin_KG(
        C[valid], phi[valid], e_grain / (2 * (1 + nu)), nu, P[valid])
    rho = rho_grain * (1 - phi)          # dry pack
    Vp = np.full(n, np.nan)
    Vs = np.full(n, np.nan)
    Vp[valid] = np.sqrt((K[valid] + 4 * G[valid] / 3) / rho[valid])
    Vs[valid] = np.sqrt(G[valid] / rho[valid])
    return dict(C=C, phi=phi, P=P, K=K, G=G, rho=rho,
                Vp=Vp, Vs=Vs, VpVs=Vp / Vs, valid=valid)


if __name__ == "__main__":
    import time
    t0 = time.time()
    pos = np.loadtxt("./txt/m4_1_pos.txt")
    rad = np.loadtxt("./txt/init_rad.txt")
    out = run(pos, rad)
    v = out["valid"]
    print(f"  particles {len(pos)}, valid (interior) {v.sum()} ({100 * v.mean():.1f}%)")
    print(f"  coordination number : mean {out['C'][v].mean():.2f}")
    print(f"  porosity            : mean {out['phi'][v].mean():.4f}")
    print(f"  pressure            : mean {out['P'][v].mean():.4g} Pa")
    print(f"  Vp {out['Vp'][v].mean():.0f} m/s   Vs {out['Vs'][v].mean():.0f} m/s   "
          f"Vp/Vs {out['VpVs'][v].mean():.4f} (std {out['VpVs'][v].std():.1e})")
    print(f"  elapsed {time.time() - t0:.0f} s")
