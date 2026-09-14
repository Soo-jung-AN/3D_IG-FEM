"""Are the synthesized seismic properties plausible, judged against the
DEM's own published constants?

An & So (2026), Comms. Earth & Environ., Supplementary Table 1 gives
density, Young's modulus and friction angle per layer for models M1-M5.
That removes the guesswork: this model's seismic properties follow from
its own elastic constants, with Poisson's ratio the only free parameter.

  Vp = sqrt( E(1-nu) / (rho (1+nu)(1-2nu)) ),  Vs = sqrt( E / (2 rho (1+nu)) )

This script reports the reference state those constants imply, checks it
against independent rock-physics trends and against real crustal values,
carries it through the deformation and the imaging chain, and contrasts
it with the two earlier calibrations.

Usage:  python3 paper_calibration.py
"""
import numpy as np
import skfmm
from scipy.spatial import cKDTree

import compare_3d as C
import rock_physics as R
import hertz_mindlin_3d as H

NU = R.NU_DEM
SEISMIC_FREQ = 5.0        # Hz -- what a 250 m grid supports (see plausibility_3d.py)
OUT = "./results/paper_calibration.npz"

# published ranges for a real 15 km column, for the verdict column
REAL = {"upper":       (3.0, 5.0, 2200, 2500),
        "middle":      (4.5, 5.8, 2500, 2700),
        "lower":       (5.8, 6.4, 2650, 2800),
        "decollement": (4.0, 5.0, 2100, 2200)}   # evaporite, the usual weak-layer rock


def verdict(lo, hi, v):
    return "ok" if lo <= v <= hi else ("LOW" if v < lo else "HIGH")


def main():
    X0 = np.loadtxt(C.UNDEFORMED)
    X1 = np.loadtxt(C.DEFORMED)
    rad = np.loadtxt(C.RADIUS)
    D = np.load(C.OUT)
    vol = D["vol_igfem"]
    gx, gy, gz = D["gx"], D["gy"], D["gz"]
    jy, dx, inside = int(D["slice_y"]), float(D["dx"]), D["inside"]

    print("=" * 86)
    print("REFERENCE STATE FROM SUPPLEMENTARY TABLE 1  (M4, nu = %.2f)" % NU)
    print("=" * 86)
    print(f"{'layer':>12} {'depth km':>11} {'rho':>6} {'E GPa':>6} {'UCS':>7} {'E/UCS':>6} |"
          f" {'Vp m/s':>7} {'Vs':>6} {'Vp/Vs':>6} {'Z e6':>7} | {'Vp':>4} {'rho':>4}")
    rows = R.paper_layer_summary(NU)
    for r in rows:
        lo, hi, rlo, rhi = REAL[r["name"]]
        print(f"{r['name']:>12} {r['z'][1]:6.0f}..{r['z'][0]:<4.0f} {r['rho']:6.0f} {r['E']:6.0f}"
              f" {r['UCS']:7.1f} {r['mod_ratio']:6.0f} | {r['Vp']:7.0f} {r['Vs']:6.0f}"
              f" {r['VpVs']:6.3f} {r['Z']/1e6:7.2f} | {verdict(lo*1e3, hi*1e3, r['Vp']):>4}"
              f" {verdict(rlo, rhi, r['rho']):>4}")

    print("\nIndependent checks on the table's own (E, rho) pairs:")
    print(f"  {'layer':>12} {'Gardner rho':>12} {'vs table':>9} | {'Brocher Vs':>11} {'vs elastic Vs':>14}")
    for r in rows:
        vp_kms = r["Vp"] / 1000.0
        g = 1741.0 * vp_kms ** 0.25
        b = R.vs_from_vp_brocher(vp_kms) * 1000.0
        print(f"  {r['name']:>12} {g:12.0f} {100*(r['rho']/g-1):+8.1f}% | {b:11.0f}"
              f" {100*(r['Vs']/b-1):+13.1f}%")
    print("  (Gardner 1974 and Brocher 2005 are fits to real rock; they never saw this model.)")

    print("\nNormal-incidence reflection coefficients of the undeformed layer stack:")
    order = [r for r in rows if r["name"] != "decollement"][::-1]     # upper -> lower
    for a, b in zip(order[:-1], order[1:]):
        Ra, Rb = a["Z"], b["Z"]
        print(f"  {a['name']:>6} / {b['name']:<12} R = {(Rb-Ra)/(Rb+Ra):+.3f}")
    dec = [r for r in rows if r["name"] == "decollement"][0]
    low = [r for r in rows if r["name"] == "lower"][0]
    print(f"  {'lower':>6} / {'decollement':<12} R = {(dec['Z']-low['Z'])/(dec['Z']+low['Z']):+.3f}"
          f"   <-- a bright, polarity-reversed reflector")

    # ---------------- deformed state ---------------------------------
    print("\n" + "=" * 86)
    print("AFTER DEFORMATION  (IG-FEM strain, m4_1)")
    print("=" * 86)
    Z, Xc = X0[:, 2], X0[:, 0]
    seed = (Z >= -15e3) & (Z < -13e3) & (Xc >= 30e3) & (Xc <= 120e3)
    rho_p, Vp_p, Vs_p = R.synthesize_paper(vol, X0, NU)
    bands = (("upper", (Z >= -7e3) & (Z < 0)),
             ("middle", (Z >= -11e3) & (Z < -7e3)),
             ("lower", (Z >= -15e3) & (Z < -11e3) & ~seed),
             ("decollement", seed))
    print(f"{'layer':>12} {'Vp km/s':>21} {'rho kg/m3':>19} {'Z e6':>8}")
    for name, m in bands:
        print(f"{name:>12}  {Vp_p[m].min():5.2f} {Vp_p[m].mean():5.2f} {Vp_p[m].max():5.2f}"
              f"   {rho_p[m].min():6.0f} {rho_p[m].mean():6.0f} {rho_p[m].max():6.0f}"
              f"   {(rho_p[m]*Vp_p[m]*1e3).mean()/1e6:7.2f}")
    print(f"  Vp/Vs is {np.sqrt(2*(1-NU)/(1-2*NU)):.4f} everywhere -- fixed by nu, which the table "
          "does not give.")

    # ---------------- Hertz-Mindlin with the table's E ----------------
    print("\n" + "=" * 86)
    print("HERTZ-MINDLIN, re-run with the table's layer moduli instead of quartz")
    print("=" * 86)
    rho_l, E_l, _ = R.paper_layer_fields(X0)
    hm = H.run(X1, rad, e_grain=E_l, rho_grain=rho_l / (1 - 0.02), verbose=False)
    v = hm["valid"]
    print(f"  P mean {hm['P'][v].mean():.3g} Pa   Vp mean {hm['Vp'][v].mean():.0f} m/s")
    for name, m in bands[:3]:
        mm = m & v
        if mm.sum():
            ref = [r for r in rows if r["name"] == name][0]["Vp"]
            print(f"  {name:>12}  HM {hm['Vp'][mm].mean():6.0f} m/s   vs elastic {ref:6.0f} m/s"
                  f"   ({100*(hm['Vp'][mm].mean()/ref-1):+5.1f}%)   n={mm.sum()}")
    print("  Hertz-Mindlin models an UNBONDED pack: stiffness from contacts at the confining")
    print("  pressure only. This DEM is bonded (M4: 60/54/48 MPa cohesion), and the bonds carry")
    print("  most of the modulus, so HM has to come out low for a cemented rock.")

    # ---------------- imaging chain ----------------------------------
    print("\n" + "=" * 86)
    print(f"EIKONAL + {SEISMIC_FREQ:.0f} Hz SECTION")
    print("=" * 86)
    axes, shape, nodes = C.make_grid(X1, dx)
    tree = cKDTree(X1)
    src = [np.argmin(np.abs(gx - 0.5 * (gx[0] + gx[-1]))),
           np.argmin(np.abs(gy - 0.5 * (gy[0] + gy[-1]))), 0]
    src[2] = np.where(inside[src[0], src[1]])[0].max()
    lsf = np.ones(shape)
    lsf[tuple(src)] = -1.0
    surf = np.array([np.where(inside[i, jy])[0].max() if inside[i, jy].any() else -1
                     for i in range(len(gx))])
    ok = surf >= 0
    ix = np.arange(len(gx))[ok]

    phi0, rg0, vp0 = R.zoned_initial_properties(X0)
    _, rho_0, Vp_0, _, _ = R.synthesize_vpvs(vol, phi0, rg0, vp0)

    store = {}
    for key, Vp_kms, rho in (("published", Vp_0, rho_0), ("paper", Vp_p, rho_p)):
        VP = C.interpolate(tree, Vp_kms * 1000.0, nodes, shape)
        RHO = C.interpolate(tree, rho, nodes, shape)
        T = skfmm.travel_time(lsf, VP, dx=dx)
        t = T[ix, jy, surf[ok]]
        s = C.section(VP[:, jy, :], RHO[:, jy, :], inside[:, jy, :], dx, SEISMIC_FREQ)
        store[f"vp_{key}"], store[f"rho_{key}"] = VP, RHO
        store[f"tt_{key}"], store[f"sec_{key}"] = T, s
        print(f"  {key:>10}  Vp {VP[inside].mean():6.0f} m/s   Z {(VP*RHO)[inside].mean()/1e6:5.2f}e6"
              f"   surface arrival mean {t.mean():5.2f} s   max {t.max():5.2f} s")
    print(f"  {'':>10}  a straight 6.0 km/s ray over the same 82 km offset: 13.7 s")
    a, b = store["sec_published"], store["sec_paper"]
    m = np.isfinite(a) & np.isfinite(b)
    print(f"\n  section corr, published vs paper at {SEISMIC_FREQ:.0f} Hz: {np.corrcoef(a[m], b[m])[0,1]:+.3f}")

    np.savez_compressed(OUT, gx=gx, gy=gy, gz=gz, dx=dx, inside=inside, slice_y=jy,
                        src=np.array([gx[src[0]], gy[src[1]], gz[src[2]]]),
                        hm_Vp=hm["Vp"], hm_valid=hm["valid"], **store)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
