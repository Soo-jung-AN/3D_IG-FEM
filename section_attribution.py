"""How much of the synthetic seismic section is the strain, and how much
is the reference model both Botter routes were given?

compare_3d.py reports a +0.971 correlation between the SSPX and IG-FEM
synthetic sections, against +0.011 for the same pair in 2D. Most of that
jump is not better agreement about the strain: the 3D reference state
(rock_physics.ZONES) puts 2.0 / 3.0 / 4.0 km/s layers into the model
before any strain is applied, and both routes get exactly the same ones,
whereas the 2D setup used a smooth linear depth trend with no contrasts
for the wavelet to reflect off.

This script builds the section of the zero-strain reference model and
separates the two contributions.
"""
import numpy as np
from scipy.spatial import cKDTree

import compare_3d as C
from rock_physics import synthesize_vpvs, zoned_initial_properties


def main():
    D = np.load(C.OUT)
    gx, gy, gz = D["gx"], D["gy"], D["gz"]
    jy, dx = int(D["slice_y"]), float(D["dx"])
    inside, secs = D["inside"][:, jy, :], D["secs"]

    X0 = np.loadtxt(C.UNDEFORMED)
    X1 = np.loadtxt(C.DEFORMED)
    phi0, rho_g, Vp0 = zoned_initial_properties(X0)
    _, rho_ref, Vp_ref, _, _ = synthesize_vpvs(np.zeros(len(X0)), phi0, rho_g, Vp0)

    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    nodes = np.column_stack([GX.ravel(), np.full(GX.size, gy[jy]), GZ.ravel()])
    tree = cKDTree(X1)
    s_ref = C.section(C.interpolate(tree, Vp_ref * 1000.0, nodes, GX.shape),
                      C.interpolate(tree, rho_ref, nodes, GX.shape),
                      inside, dx, C.SEISMIC_FREQ)

    m = np.isfinite(s_ref) & np.isfinite(secs[0]) & np.isfinite(secs[1])
    a, b, r = secs[0][m], secs[1][m], s_ref[m]
    resid = lambda v: v - np.polyval(np.polyfit(r, v, 1), r)
    ra, rb = resid(a), resid(b)

    print(f"n = {m.sum()} live samples on the y = {gy[jy]/1000:.1f} km slice\n")
    print(f"  vs the zero-strain reference section:  SSPX {np.corrcoef(a, r)[0,1]:+.3f}"
          f"   IG-FEM {np.corrcoef(b, r)[0,1]:+.3f}")
    print(f"  SSPX vs IG-FEM, raw                 :  {np.corrcoef(a, b)[0,1]:+.3f}")
    print(f"  SSPX vs IG-FEM, reference removed   :  {np.corrcoef(ra, rb)[0,1]:+.3f}")
    print(f"  strain-driven share of the amplitude:  SSPX {np.std(ra)/np.std(a):.3f}"
          f"   IG-FEM {np.std(rb)/np.std(b):.3f}")


if __name__ == "__main__":
    main()
