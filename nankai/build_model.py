"""Build and run the Nankai wedge in PFC3D.

RUNS INSIDE PFC. Everything geometric lives in model_spec.py, which has
no `itasca` in it and is tested outside; this file is the thin layer that
turns that into PFC commands, and it is the part that has NOT been run
against a PFC installation. Treat the command strings as a first draft:
the syntax below is PFC 6/7-style (`model`, `ball distribute`,
`contact cmat`), and if your build disagrees the fix is local to the
`it.command(...)` blocks.

    pfc3d  call  nankai/build_model.py

or through the pipeline, which also runs the analysis afterwards:

    python3 pfc_pipeline.py nankai/build_model.py --stem runs/nankai01 \
        --alpha 200 --zmax -100

Sequence:

  1. domain and gravity, rotated so the décollement is flat
  2. fill the prism envelope with balls, delete what falls outside
  3. group by unit, assign the linear parallel bond model per unit
  4. weaken the seeded fault zones
  5. settle under gravity, then export the REFERENCE state
  6. drive the conveyor landward, export the DEFORMED state

The reference export happens after settling, not before: the strain we
want is the increment caused by convergence, not the pack's own initial
compaction.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itasca as it                                    # noqa: E402  (PFC only)

import pfc_export                                      # noqa: E402
from nankai.model_spec import (Model, PROPERTIES, UNITS, UNIT_NAMES,
                               WEAK_FACTOR, KM)         # noqa: E402

# ---- run parameters --------------------------------------------------
R_MEAN, R_RATIO = 33.0, 1.5        # m; 269k particles at slab = 264 m
SLAB = 264.0                        # m, 4 mean diameters: this is plane strain
SHORTENING = 3000.0                 # m of convergence to impose
PLATE_VELOCITY = 1.0                # m/step in model time; quasi-static, not real
STEM = os.environ.get("NANKAI_STEM", "runs/nankai01")
BASAL_FRICTION = float(os.environ.get("NANKAI_BASAL_FRICTION", "0.15"))

m = Model(r_mean=R_MEAN, r_ratio=R_RATIO, slab=SLAB)


def cmd(s):
    it.command(s)


def build():
    gx, gy, gz = m.gravity()
    x_lo, x_hi = m.x0 * KM, m.x1 * KM
    z_lo = -m.envelope(np.array([m.x1]))[1][0] * KM - 500.0
    z_hi = -m.envelope(np.array([m.x0]))[0][0] * KM + 500.0
    # the envelope is expressed in the data frame; the domain has to hold
    # it after rotation, so take the rotated corners
    cx, cz = m.to_model(np.array([x_lo, x_lo, x_hi, x_hi]),
                        np.array([z_lo, z_hi, z_lo, z_hi]))

    cmd("model new")
    cmd("model title 'Nankai accretionary wedge'")
    cmd(f"model domain extent {cx.min()-1000:.0f} {cx.max()+1000:.0f} "
        f"0 {SLAB:.0f} {cz.min()-1000:.0f} {cz.max()+1000:.0f} condition destroy")
    cmd(f"model gravity {gx:.6f} {gy:.6f} {gz:.6f}")
    cmd("model random 10101")
    cmd("contact cmat default model linearpbond method deform emod 1e9 kratio 1.5")

    # fill the bounding box, then delete everything outside the envelope
    cmd(f"ball distribute porosity 0.38 radius {R_MEAN/R_RATIO**0.5:.3f} "
        f"{R_MEAN*R_RATIO**0.5:.3f} "
        f"box {cx.min():.0f} {cx.max():.0f} 0 {SLAB:.0f} {cz.min():.0f} {cz.max():.0f}")
    print(f"[build] {it.ball.count()} balls before trimming", flush=True)
    trim()
    print(f"[build] {it.ball.count()} balls inside the envelope "
          f"(model_spec predicted {m.n_particles():,})", flush=True)


def ball_positions():
    """(n, 3) ball centres, vectorised where the build offers it."""
    try:
        from itasca import ballarray as ba
        return np.asarray(ba.pos(), dtype=float)
    except (ImportError, AttributeError):
        return np.array([b.pos() for b in it.ball.list()], dtype=float)


def data_frame_xz(P=None):
    """Ball positions rotated back into the data frame, which is where
    model_spec does all of its classification."""
    P = ball_positions() if P is None else P
    return m.to_data(P[:, 0], P[:, 2])


def trim():
    """Delete everything outside the prism envelope. Marking a group and
    deleting it in one command is far cheaper than deleting by id."""
    x, z = data_frame_xz()
    kill = ~m.inside(x, z)
    if not kill.any():
        return
    for b, k in zip(it.ball.list(), kill):
        if k:
            b.set_group("outside", "trim")
    cmd("ball delete range group 'outside' slot 'trim'")


def assign_units():
    P = _ball_xyz()
    x, z = _data_frame_xz(P)
    u = m.unit(x, z)
    weak = m.weak(x, z)
    conv = m.conveyor(x, z)

    for b, code, w, c in zip(it.ball.list(), u, weak, conv):
        b.set_group(UNIT_NAMES[int(code)], slot="unit")
        b.set_group("weak" if w else "intact", slot="fault")
        b.set_group("conveyor" if c else "free", slot="drive")

    for name, p in PROPERTIES.items():
        n = int((u == UNITS[name]).sum())
        if n == 0:
            continue
        fric = BASAL_FRICTION if name == "decollement" else p["friction"]
        cmd(f"ball attribute density {p['density']:.1f} "
            f"range group '{name}' slot 'unit'")
        # intact rock, then the same rock inside a seeded fault. Applied
        # as two explicit values rather than a multiplier, because not
        # every PFC build accepts `multiply` on contact property.
        for tag, f in (("intact", 1.0), ("weak", WEAK_FACTOR)):
            cmd(f"contact property "
                f"pb_ten {p['tensile'] * f:.6g} pb_coh {p['cohesion'] * f:.6g} "
                f"fric {fric:.3f} pb_deform emod {p['E']:.6g} kratio 1.5 "
                f"range group '{name}' slot 'unit' group '{tag}' slot 'fault'")
        print(f"[units] {name:<12} {n:7,}  fric {fric:.2f}  E {p['E']:.3g}", flush=True)
    print(f"[units] seeded faults bonded at {WEAK_FACTOR:.2f} of host strength",
          flush=True)


def settle():
    cmd("model cycle 1 calm")
    cmd("contact method bond gap 0.0")
    cmd("model solve ratio 1e-5")
    cmd("ball attribute displacement 0 0 0")           # zero the datum
    print("[settle] equilibrated under tilted gravity", flush=True)


def drive():
    cmd("ball fix velocity range group 'conveyor' slot 'drive'")
    cmd(f"ball attribute velocity-x {PLATE_VELOCITY:.4g} "
        f"range group 'conveyor' slot 'drive'")
    steps = int(SHORTENING / PLATE_VELOCITY)
    print(f"[drive] {SHORTENING:.0f} m of convergence in {steps} steps", flush=True)
    cmd(f"model cycle {steps}")


if __name__ == "__main__":
    build()
    assign_units()
    settle()
    pfc_export.export(STEM, reference=True)
    drive()
    pfc_export.export(STEM)
    print(f"[done] exported to {STEM}_*.txt", flush=True)
