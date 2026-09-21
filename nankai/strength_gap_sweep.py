"""Map bond strength against bonding gap on one relaxed pack.

Ran once to answer whether a BONDED Nankai wedge can be made to stand up.
It cannot at any defensible strength, which is why build_model defaults to
NANKAI_WEDGE=frictional; the script is kept because the grid is the
evidence for that decision and because the same sweep is the way to test
any future change to model_spec.PROPERTIES.

    NANKAI_R_MEAN=200 NANKAI_SLAB=1600 NANKAI_SOLVE_CYCLES=3000 \
    NANKAI_GRAVITY_STEPS=4 NANKAI_GRAVITY_STEP_CYCLES=200 \
    python3 pfc_pipeline.py nankai/strength_gap_sweep.py \
        --stem runs/grid --only pfc
    python3 nankai/strength_gap_plot.py <grid.csv> <out.png>

The sweep writes grid.csv next to this file; the second command
draws it. They are separate because the sweep only runs inside PFC
and matplotlib is not wanted there.

build() and relax() cost the same for every combination, so they are paid
once: the relaxed pack is saved and restored before each point. The CMAT
proximity is set from the LARGEST gap in the grid, so every contact any
combination might bond exists in all of them.

NOT an equilibrium study. Every point stops at the same cycle cap, so the
points are comparable with each other even though none of them has
settled -- see build_model.report_settling.

What it found, on a 6,627-ball pack:
  - bonding gap moves the bonded fraction (43% at 0.05 r_min, 62% at
    0.40) and barely moves alpha_0 at all.
  - strength moves everything. x1 to x10 leaves alpha_0 about 1.0-1.4 deg
    off the section; x100 drops it to +0.24 with no tensile failures and
    no particles lost.
  - x100 means pb_ten of 100 MPa in the outer prism, harder than granite.
    That is the result: a bonded wedge needs an indefensible cement.
"""

import collections
import os
import sys
import time

import itasca as it

it.command("python-reset-state false")   # or `model restore` wipes this script

from nankai import build_model as B
from nankai import taper

STRENGTHS = [1.0, 3.0, 10.0, 30.0, 100.0]
GAPS = [0.05, 0.10, 0.20, 0.40]
SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relaxed")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid.csv")

BASE = {k: dict(v) for k, v in B.PROPERTIES.items()}


def set_strength(mult):
    for k, v in BASE.items():
        B.PROPERTIES[k]["tensile"] = v["tensile"] * mult
        B.PROPERTIES[k]["cohesion"] = v["cohesion"] * mult


def states():
    st = collections.Counter()
    for c in B.contacts():
        st[c.prop("pb_state")] += 1
    return st


# ---- the pack, once -------------------------------------------------
B.BOND_GAP = max(GAPS) * B.R_MIN          # proximity must cover every gap
B.build()
B.group_balls()
B.assign_densities()
B.hold_boundaries()
B.relax()
n_built = it.ball.count()
it.command("model save '%s'" % SAVE.replace("\\", "/"))
print("[sweep] relaxed pack saved: %d balls, proximity %.2f m"
      % (n_built, B.BOND_GAP), flush=True)

with open(OUT, "w") as f:
    f.write("strength,gap_frac,gap_m,bonded_at_bond,bonded_after,"
            "tension,shear,never,n_contacts,lost,alpha0,alpha_target,seconds\n")

target = taper.target_alpha()
for mult in STRENGTHS:
    for gf in GAPS:
        t0 = time.time()
        it.command("model restore '%s'" % SAVE.replace("\\", "/"))
        set_strength(mult)
        B.BOND_GAP = gf * B.R_MIN
        B.assign_contacts()
        at_bond = B.bond()
        B.apply_gravity()
        B.settle()
        st = states()
        tot = sum(st.values())
        a0, _, _ = taper.measure(B.ball_positions(), B.m.beta)
        lost = n_built - it.ball.count()
        row = (mult, gf, B.BOND_GAP, at_bond, st[3], st[1], st[2], st[0],
               tot, lost, a0, target, time.time() - t0)
        with open(OUT, "a") as f:
            f.write(",".join("%.6g" % v for v in row) + "\n")
        print("[sweep] x%-6g gap %.2f -> bonded %5.1f%% (was %5.1f%%)  "
              "tension %5.1f%%  shear %4.1f%%  lost %5d  alpha0 %7.3f "
              "(err %+6.3f)  %4.0fs"
              % (mult, gf, 100.0 * st[3] / tot, 100.0 * at_bond / tot,
                 100.0 * st[1] / tot, 100.0 * st[2] / tot, lost, a0,
                 a0 - target, time.time() - t0), flush=True)

print("SWEEP_DONE -> %s" % OUT, flush=True)
it.command("program quit")
