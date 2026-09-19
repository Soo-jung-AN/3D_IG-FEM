"""Is the synthesized rock physics physically plausible for THIS model?

compare_3d.py answers "do the three routes agree". This answers the prior
question: are the numbers they agree on the right size for a 15 km
crustal column under extension?

Three things are checked against the model's own assumptions and against
standard crustal relations (Brocher, 2005, BSSA 95, 2081-2092):

  1. Vp/Vs and Poisson's ratio. Eq. (4) of Botter et al. is Han's (1986)
     water-saturated SANDSTONE fit, valid over roughly 3.0-5.5 km/s.
  2. Bulk density. ZONES hands Eq. (2) the DEM's own densities as GRAIN
     densities and then mixes water in, so the synthesized rock is
     lighter than the rock the DEM computed its stresses for.
  3. Impedance and velocity level, against a real crustal column.

It then rebuilds the whole chain -- Vp, rho, 3D eikonal traveltime,
synthetic section -- with rock_physics.crustal_initial_properties(),
which ties Vp_ini to the DEM's own density structure through the
Nafe-Drake curve, and reports what changes.

Usage:  python3 plausibility_3d.py
"""
import numpy as np
import skfmm
from scipy.spatial import cKDTree

import compare_3d as C
import rock_physics as R

FREQS = (30.0, 5.0)          # as run before, and what the grid supports
OUT = "./results/plausibility_3d.npz"


def poisson(vp_vs):
    return (vp_vs ** 2 - 2) / (2 * (vp_vs ** 2 - 1))


def band(Z):
    return (("  0 to  -7 km", (Z >= -7e3) & (Z < 0)),
            (" -7 to -11 km", (Z >= -11e3) & (Z < -7e3)),
            ("-11 to -15 km", (Z >= -15e3) & (Z < -11e3)))


def audit(tag, phi, rho, Vp, Vs, Z, dem_rho):
    print(f"\n--- {tag} " + "-" * (58 - len(tag)))
    r = Vp / Vs
    nu = poisson(r)
    print(f"  Vp/Vs   p5 {np.percentile(r,5):6.3f}  median {np.median(r):6.3f}  p95 {np.percentile(r,95):7.3f}"
          f"  max {r.max():7.3f}")
    print(f"  Poisson p5 {np.percentile(nu,5):6.3f}  median {np.median(nu):6.3f}  p95 {np.percentile(nu,95):7.3f}")
    print(f"  Poisson > 0.40 (near-fluid): {100*np.mean(nu>0.40):5.1f}% of particles")
    print(f"  Vp outside Han's 3.0-5.5 km/s calibration range: {100*np.mean((Vp<3.0)|(Vp>5.5)):5.1f}%")
    for lab, m in band(Z):
        print(f"  {lab}  phi {phi[m].mean():.3f}   bulk rho {rho[m].mean():6.0f} kg/m3"
              f"  (DEM assumes {dem_rho[m].mean():6.0f}, {100*(rho[m].mean()/dem_rho[m].mean()-1):+5.1f}%)"
              f"   Vp {Vp[m].mean():4.2f} km/s   Z {rho[m].mean()*Vp[m].mean()/1e3:5.2f}e6")


def main():
    D = np.load(C.OUT)
    gx, gy, gz = D["gx"], D["gy"], D["gz"]
    jy, dx, inside = int(D["slice_y"]), float(D["dx"]), D["inside"]
    vol = D["vol_igfem"]

    X0 = np.loadtxt(C.UNDEFORMED)
    X1 = np.loadtxt(C.DEFORMED)
    Z = X0[:, 2]

    # the bulk densities the DEM itself used for lithostatic stress
    dem_rho = np.full(len(Z), 2300.0)
    dem_rho[(Z >= -11e3) & (Z < -7e3)] = 2500.0
    dem_rho[(Z >= -15e3) & (Z < -11e3)] = 2700.0
    dem_rho[(Z >= -15e3) & (Z < -13e3) & (X0[:, 0] >= 30e3) & (X0[:, 0] <= 120e3)] = 2100.0

    print("=" * 72)
    print("PLAUSIBILITY AUDIT -- IG-FEM strain, m4_1 stage, 259,943 particles")
    print("=" * 72)

    cases = {}
    for tag, props, vs_fn in (
            ("as published: ZONES + Han (1986)", R.zoned_initial_properties, R.vs_from_vp),
            ("crustal: Nafe-Drake Vp + Brocher Vs", R.crustal_initial_properties, R.vs_from_vp_brocher)):
        phi0, rho_g, Vp0 = props(X0)
        phi = R.porosity_from_strain(vol, phi0)
        rho = R.density_from_porosity(phi, rho_g)
        Vp = R.vp_from_strain(vol, Vp0)
        Vs = vs_fn(Vp)
        audit(tag, phi, rho, Vp, Vs, Z, dem_rho)
        cases[tag] = (phi, rho, Vp * 1000.0, Vs * 1000.0)

    print("\n  reference, a real 15 km crustal column (Brocher 2005; Christensen & Mooney 1995):")
    print("    basin fill 0-7 km      Vp 2.5-4.5 km/s   rho 2200-2500   Z  5.5-11.3e6")
    print("    deep basin 7-11 km     Vp 4.5-5.5 km/s   rho 2500-2650   Z 11.3-14.6e6")
    print("    upper crust 11-15 km   Vp 5.8-6.3 km/s   rho 2700-2800   Z 15.7-17.6e6")

    print("\n" + "=" * 72)
    print("WHAT THE 250 m GRID CAN CARRY")
    print("=" * 72)
    for f in (30.0, 20.0, 10.0, 5.0, 2.5):
        for v, lab in ((2700.0, "current Vp"), (5000.0, "crustal Vp")):
            q = v / f / 4.0
            print(f"  {f:4.1f} Hz, {lab} {v/1000:.1f} km/s: lambda/4 = {q:5.0f} m "
                  f"{'<' if q < dx else '>='} {dx:.0f} m grid  "
                  f"{'-- resolves nothing the model contains' if q < dx else '-- OK'}")
    print(f"\n  the grid resolves lambda/4 = {dx:.0f} m, so the honest upper frequency is")
    print(f"  f = Vp / (4 dx): {2700/(4*dx):.1f} Hz at 2.7 km/s, {5000/(4*dx):.1f} Hz at 5.0 km/s.")

    # ---- rebuild the chain with the crustal calibration -------------
    print("\n" + "=" * 72)
    print("RE-RUN: eikonal traveltime and synthetic section, both calibrations")
    print("=" * 72)
    axes, shape, nodes = C.make_grid(X1, dx)
    tree = cKDTree(X1)
    src = [np.argmin(np.abs(gx - 0.5 * (gx[0] + gx[-1]))),
           np.argmin(np.abs(gy - 0.5 * (gy[0] + gy[-1]))), 0]
    src[2] = np.where(inside[src[0], src[1]])[0].max()
    phi_lsf = np.ones(shape)
    phi_lsf[tuple(src)] = -1.0

    surf = np.array([np.where(inside[i, jy])[0].max() if inside[i, jy].any() else -1
                     for i in range(len(gx))])
    ok = surf >= 0
    ix = np.arange(len(gx))[ok]

    store = {}
    for tag, (phi, rho, Vp, Vs) in cases.items():
        VP = C.interpolate(tree, Vp, nodes, shape)
        RHO = C.interpolate(tree, rho, nodes, shape)
        T = skfmm.travel_time(phi_lsf, VP, dx=dx)
        t_surf = T[ix, jy, surf[ok]]
        print(f"\n  {tag}")
        print(f"    gridded Vp mean {VP[inside].mean():6.0f} m/s   impedance mean "
              f"{(VP*RHO)[inside].mean()/1e6:5.2f}e6")
        print(f"    surface first arrival: mean {t_surf.mean():6.2f} s   max {t_surf.max():6.2f} s "
              f"at {abs(gx[ix[t_surf.argmax()]]-gx[src[0]])/1000:.0f} km offset")
        key = "crustal" if tag.startswith("crustal") else "published"
        store[f"vp_{key}"], store[f"rho_{key}"], store[f"tt_{key}"] = VP, RHO, T
        for f in FREQS:
            s = C.section(VP[:, jy, :], RHO[:, jy, :], inside[:, jy, :], dx, f)
            store[f"sec_{key}_{int(f)}"] = s
            fin = np.isfinite(s)
            print(f"    section {f:4.1f} Hz: RMS amplitude {np.sqrt(np.mean(s[fin]**2)):.2e}")

    for f in FREQS:
        a, b = store[f"sec_published_{int(f)}"], store[f"sec_crustal_{int(f)}"]
        m = np.isfinite(a) & np.isfinite(b)
        print(f"\n  section corr, published vs crustal at {f:4.1f} Hz: {np.corrcoef(a[m], b[m])[0,1]:+.3f}")

    np.savez_compressed(OUT, gx=gx, gy=gy, gz=gz, dx=dx, inside=inside, slice_y=jy,
                        src=np.array([gx[src[0]], gy[src[1]], gz[src[2]]]), **store)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
