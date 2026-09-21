"""Calibrate basal friction against the observed taper.

The wedge starts at the observed geometry, so the taper is right at step
zero by construction. What basal friction controls is whether it STAYS
right: too much and the wedge thickens and steepens, too little and it
spreads and shallows. So the sweep runs a short convergence increment at
each friction and measures d(alpha), and the calibrated value is where
d(alpha) crosses zero.

    python3 nankai/friction_sweep.py --run 0.10 0.15 0.20 0.25 0.30
    python3 nankai/friction_sweep.py --analyse            # measure what exists
    python3 nankai/friction_sweep.py --demo               # no PFC needed

--run launches PFC once per value with NANKAI_BASAL_FRICTION set, into
runs/sweep_mu<value>_*.txt. The PFC invocation comes from the same
pfc_pipeline.json as the main pipeline, via pfc_pipeline.pfc_argv, so
the PFC-6 .dat wrapper is built the same way in both places.

The runs are sequential and each is a full model, so a five-value sweep
is a long job. Set NANKAI_R_MEAN (and NANKAI_SLAB with it) to sweep at a
coarser resolution first if you only want the sign of d(alpha).
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pfc_pipeline
from nankai import taper
from nankai.model_spec import KM, Model

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
SHORT_SHORTENING = 800.0        # m -- enough to see the surface respond

# alpha_0 is the surface slope BEFORE any convergence. Every run is built
# from the same digitised section, so alpha_0 is the same number in every
# run by construction -- unless the settling stage is moving the surface,
# in which case what the sweep measures is settling, not friction. This
# is the gate: until the spread in alpha_0 across the swept friction
# values is below it, report() refuses to name a calibrated mu_b.
ALPHA0_TOLERANCE = 0.05         # degrees


def stem_for(mu):
    return os.path.join(RUNS, f"sweep_mu{mu:.3f}")


def launch(mu):
    cfg = pfc_pipeline.load_config()
    exe = os.environ.get("PFC_EXE") or cfg.get("pfc_exe")
    if not exe:
        raise SystemExit("PFC executable not known: set PFC_EXE or write "
                         "pfc_pipeline.json (see pfc_pipeline.py).")
    script = os.path.join(HERE, "build_model.py")
    env = dict(os.environ, NANKAI_BASAL_FRICTION=f"{mu:.4f}",
               NANKAI_STEM=stem_for(mu),
               NANKAI_SHORTENING=f"{SHORT_SHORTENING:.1f}")
    cmd = pfc_pipeline.pfc_argv(exe, cfg, script)
    print(f"\n=== mu_b = {mu:.3f} ===\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"PFC failed at mu_b = {mu:.3f} (exit {r.returncode})")
    # PFC quits 0 even when the Python it ran raised, so the return code
    # is not evidence that anything was written. Check the export.
    if load(mu) is None:
        raise SystemExit(
            f"PFC returned 0 at mu_b = {mu:.3f} but wrote no export to "
            f"{stem_for(mu)}_*.txt -- a Python traceback inside PFC does "
            f"not set the exit code, so read the PFC output above for it.")


def load(mu):
    s = stem_for(mu)
    i, p = f"{s}_init_pos.txt", f"{s}_pos.txt"
    if not (os.path.exists(i) and os.path.exists(p)):
        return None
    return np.loadtxt(i), np.loadtxt(p)


def report(rows):
    at = taper.target_alpha()
    print(f"\ntarget: alpha {at:.3f} deg, measured the same way as the runs\n")
    print(f"{'mu_b':>6} {'alpha_0':>9} {'alpha_1':>9} {'d_alpha':>9} "
          f"{'taper_1':>9} {'verdict':>22}")
    for mu, r in rows:
        # ASCII only: this line is printed, and a Windows console on a
        # Korean locale is cp949, which cannot encode an em dash --
        # the report died with UnicodeEncodeError after a sweep had
        # already spent the PFC time.
        v = ("steepening - friction too high" if r["d_alpha"] > 0.05 else
             "spreading - friction too low" if r["d_alpha"] < -0.05 else
             "stable  <-- calibrated")
        print(f"{mu:6.3f} {r['alpha_initial']:9.3f} {r['alpha_final']:9.3f} "
              f"{r['d_alpha']:+9.3f} {r['taper_final']:9.3f} {v:>22}")
    d = np.array([r["d_alpha"] for _, r in rows])
    mus = np.array([mu for mu, _ in rows])
    a0 = np.array([r["alpha_initial"] for _, r in rows])
    spread = float(a0.max() - a0.min()) if len(a0) > 1 else 0.0

    crossing = None
    if len(mus) > 1 and d.min() < 0 < d.max():
        k = np.argsort(mus)
        crossing = float(np.interp(0.0, d[k], mus[k]))

    print(f"\nalpha_0 spread across the sweep: {spread:.3f} deg "
          f"(tolerance {ALPHA0_TOLERANCE:.2f})")
    if spread > ALPHA0_TOLERANCE:
        print("\n  *** NOT A CALIBRATION ***")
        print("  alpha_0 is the surface slope before any convergence.")
        print("  Every run starts from the same digitised section, so it")
        print("  has to be the same number in all of them. It is not,")
        print("  which means the settling stage is moving the surface and")
        print("  d(alpha) is measuring settling, not basal friction.")
        if crossing is not None:
            print("  (d(alpha) would cross zero at mu_b = %.3f. "
                  "Do not use it.)" % crossing)
        print("  Fix settling first: build_model.report_settling prints")
        print("  the bonded fraction, the particles lost and alpha_0.")
    elif crossing is not None:
        print(f"\nd(alpha) crosses zero at mu_b = {crossing:.3f}")
        print(f"  alpha_0 agrees to {spread:.3f} deg across the sweep, "
              f"so this is a calibration.")
    return mus, d


def plot(rows, out=os.path.join(HERE, "fig_friction_sweep.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = np.array([mu for mu, _ in rows])
    d = np.array([r["d_alpha"] for _, r in rows])
    a1 = np.array([r["alpha_final"] for _, r in rows])
    at = taper.target_alpha()
    k = np.argsort(mus)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].axhline(0, color="#999", lw=1)
    ax[0].plot(mus[k], d[k], "o-", lw=2, color="#15616d")
    if d.min() < 0 < d.max():
        mu0 = np.interp(0.0, d[k], mus[k])
        ax[0].axvline(mu0, color="#b4531b", ls="--", lw=1.4)
        ax[0].text(mu0, ax[0].get_ylim()[1] * .85, f"  μ_b = {mu0:.3f}",
                   color="#b4531b", fontsize=9)
    ax[0].set_xlabel("basal friction μ_b"); ax[0].set_ylabel("Δα over the increment (°)")
    ax[0].set_title("the wedge steepens above, spreads below", fontsize=10.5)
    ax[0].grid(alpha=.25)

    ax[1].axhline(at, color="#b4531b", ls="--", lw=1.4, label="observed α")
    ax[1].plot(mus[k], a1[k], "o-", lw=2, color="#15616d", label="after the increment")
    ax[1].set_xlabel("basal friction μ_b"); ax[1].set_ylabel("surface slope α (°)")
    ax[1].set_title("surface slope against the section", fontsize=10.5)
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out, dpi=145)
    print(f"saved {out}")


def demo():
    """No PFC: fabricate two exports from the preview packing, the second
    with the surface tilted by a known amount, and check that the
    measurement recovers it. Verifies the analysis half end to end."""
    m = Model()
    px, pz = m.preview_packing()
    X, Z = m.to_model(px, pz)
    P0 = np.column_stack([X, np.zeros_like(X), Z])
    rows = []
    print("demo: surfaces tilted by a known amount, no PFC involved")
    for mu, tilt in ((0.10, -0.30), (0.15, -0.12), (0.20, +0.02),
                     (0.25, +0.18), (0.30, +0.40)):
        # rotate about the middle of the fitted window, in the data frame
        xd, zd = m.to_data(P0[:, 0], P0[:, 2])
        x0 = 15.5 * KM
        zd2 = zd + (xd - x0) * np.tan(np.radians(tilt))
        X2, Z2 = m.to_model(xd, zd2)
        P1 = np.column_stack([X2, np.zeros_like(X2), Z2])
        r = taper.compare(P0, P1, m.beta)
        rows.append((mu, r))
        print(f"  imposed {tilt:+.2f} deg -> recovered {r['d_alpha']:+.3f} deg"
              f"   (error {r['d_alpha'] - tilt:+.3f})")
    report(rows)
    plot(rows, out=os.path.join(HERE, "fig_friction_sweep_demo.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", type=float, default=None,
                    help="basal friction values to launch PFC at")
    ap.add_argument("--analyse", action="store_true",
                    help="measure whatever runs already exist in runs/")
    ap.add_argument("--demo", action="store_true", help="no PFC needed")
    a = ap.parse_args()

    if a.demo:
        return demo()

    os.makedirs(RUNS, exist_ok=True)
    if a.run:
        for mu in a.run:
            launch(mu)
    mus = a.run or sorted(
        float(f.split("_mu")[1].split("_init")[0])
        for f in os.listdir(RUNS) if f.startswith("sweep_mu") and f.endswith("_init_pos.txt"))
    rows = []
    for mu in mus:
        got = load(mu)
        if got is None:
            print(f"  mu_b = {mu:.3f}: no export found, skipping")
            continue
        rows.append((mu, taper.compare(got[0], got[1], Model().beta)))
    if not rows:
        raise SystemExit("nothing to measure. Run with --run, or --demo to test "
                         "the analysis without PFC.")
    report(rows)
    plot(rows)


if __name__ == "__main__":
    main()
