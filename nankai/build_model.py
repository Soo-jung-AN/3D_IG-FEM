"""Build and run the Nankai wedge in PFC3D.

RUNS INSIDE PFC. Everything geometric lives in model_spec.py, which has
no `itasca` in it and is tested outside; this file is the thin layer that
turns that into PFC commands. Every command string here has now been
executed against PFC 6.00 Release 008, and the whole sequence runs to a
pair of exports. What is NOT settled is the physics -- see THE OPEN
QUESTION below.

Run it through the pipeline, which also runs the analysis afterwards:

    python3 pfc_pipeline.py nankai/build_model.py --stem runs/nankai01 \
        --alpha 200 --zmax -100

Do not hand the .py to PFC yourself. PFC 6.0's console only auto-calls a
file whose extension its command processor recognises, so a bare .py
loads and then sits at the `pfc3d>` prompt forever. pfc_pipeline
generates the .dat wrapper and the shim that make it work.

Sequence:

  1. domain and gravity, rotated so the décollement is flat. y is
     PERIODIC, which is what makes the slab plane strain
  2. fill the prism envelope with balls, delete what falls outside
  3. group by unit, seeded fault and boundary role; set ball density
  4. hold the boundaries. The model has no walls, so the base and the
     landward backstop are layers of fixed particles
  5. relax the overlaps `ball distribute` leaves behind, UNBONDED
  6. only now assign per-unit contact stiffness and strength: relaxation
     destroys and recreates contacts, so anything set earlier is lost
  7. bond, settle under gravity, export the REFERENCE state
  8. drive the conveyor landward, export the DEFORMED state

The reference export happens after settling, not before: the strain we
want is the increment caused by convergence, not the pack's own initial
compaction.

THE OPEN QUESTION. The commands run; the wedge is not yet in
equilibrium. On a 6,627-ball smoke pack the settling solve stalls at
ratio-average ~2e-2 against its 1e-5 target, sheds about a third of its
particles off the free toe, and comes out at alpha 1.68 deg against the
section's 2.40 deg. Bond strengths, the bonding gap and the particle
size are the knobs, and none of them is calibrated. Do not read a
friction sweep as a calibration until the settled wedge holds the
observed taper on its own.
"""
import os
import sys

import numpy as np

# PFC 6.0's `program call` compiles a .py as an anonymous string, so
# __file__ is not defined in a script it runs directly. pfc_pipeline's
# generated shim defines one; this fallback covers a bare
# `pfc3d> call nankai/build_model.py`, where the repository is the
# working directory.
try:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:                                      # PFC program call
    ROOT = os.path.abspath(os.getcwd())
sys.path.insert(0, ROOT)

import itasca as it                                    # noqa: E402  (PFC only)

import pfc_export                                      # noqa: E402
from nankai.model_spec import (Model, PROPERTIES, UNITS,
                               WEAK_FACTOR, KM)         # noqa: E402

# ---- run parameters --------------------------------------------------
# R_MEAN and SLAB are environment-overridable so the command strings can
# be smoke-tested at a few thousand particles before a 269k run is
# committed to. SLAB tracks R_MEAN at 4 mean diameters, which is what
# makes the run plane strain; override both together or neither.
R_MEAN = float(os.environ.get("NANKAI_R_MEAN", "33.0"))   # m; 269k particles
R_RATIO = 1.5
SLAB = float(os.environ.get("NANKAI_SLAB", str(8.0 * R_MEAN)))   # 4 mean diameters
SHORTENING = float(os.environ.get("NANKAI_SHORTENING", "3000.0"))   # m of convergence
PLATE_VELOCITY = 1.0                # m/step in model time; quasi-static, not real
BASAL_FRICTION = float(os.environ.get("NANKAI_BASAL_FRICTION", "0.15"))
# a cap on the settling solve, so a smoke test does not sit in `model
# solve` for hours. 0 means no cap: solve to the ratio and no sooner.
SOLVE_CYCLES = int(os.environ.get("NANKAI_SOLVE_CYCLES", "0"))
# `ball distribute` hits its target porosity by OVERLAPPING particles --
# PFC says so as it runs ("There may be huge overlaps!"). Bonding that
# pack locks in the overlap forces and it blows itself apart; with
# `condition destroy` on the domain the escapees are silently deleted,
# so the symptom is a particle count that collapses between stages. The
# relaxation below cycles the unbonded pack with periodic calming to let
# the overlaps push themselves out before anything is bonded.
RELAX_CYCLES = int(os.environ.get("NANKAI_RELAX_CYCLES", "4000"))
RELAX_CALM = int(os.environ.get("NANKAI_RELAX_CALM", "10"))

# PFC's "current directory" is not the directory this file is in (it is
# the project folder, or the copied application data, or the user
# profile -- see the program call docs), so a relative stem would write
# the exports somewhere surprising. Anchor it to the repository.
STEM = os.environ.get("NANKAI_STEM", "runs/nankai01")
if not os.path.isabs(STEM):
    STEM = os.path.join(ROOT, STEM)

m = Model(r_mean=R_MEAN, r_ratio=R_RATIO, slab=SLAB)


def cmd(s):
    it.command(s)


def census(stage):
    """Balls and contacts at a stage. The domain is `condition destroy`,
    so a ball that leaves is deleted without comment -- a count that
    falls between two stages is the only warning you get, and it breaks
    the export invariant that the reference and deformed states hold the
    same particles in the same order."""
    n = it.ball.count()
    print(f"[census] {stage:<22} {n:>8,} balls  {it.contact.count():>9,} contacts",
          flush=True)
    return n


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
    # `condition` takes one keyword per direction: x, y, z. y is
    # PERIODIC, which is what makes a four-diameter slab plane strain --
    # with `destroy` in y the two faces of the slab are open and the pack
    # bleeds particles out of the sides. x and z stay `destroy`; the toe
    # and the free surface are meant to be open.
    cmd(f"model domain extent {cx.min()-1000:.0f} {cx.max()+1000:.0f} "
        f"0 {SLAB:.0f} {cz.min()-1000:.0f} {cz.max()+1000:.0f} "
        f"condition destroy periodic destroy")
    cmd(f"model gravity {gx:.6f} {gy:.6f} {gz:.6f}")
    cmd("model random 10101")
    cmd("contact cmat default model linearpbond "
        "method deformability emod 1e9 kratio 1.5")

    # fill the bounding box, then delete everything outside the envelope
    cmd(f"ball distribute porosity 0.38 radius {R_MEAN/R_RATIO**0.5:.3f} "
        f"{R_MEAN*R_RATIO**0.5:.3f} "
        f"box {cx.min():.0f} {cx.max():.0f} 0 {SLAB:.0f} {cz.min():.0f} {cz.max():.0f}")
    print(f"[build] {it.ball.count()} balls before trimming", flush=True)
    trim()
    print(f"[build] {it.ball.count()} balls inside the envelope "
          f"(model_spec predicted {m.n_particles():,})", flush=True)


def set_group_mask(mask, name, slot):
    """Put every ball where `mask` is True into `name` in `slot`.

    ballarray.set_group does the whole pack in one call. The per-ball
    fallback must pass the slot POSITIONALLY: PFC 6.0's Ball.set_group is
    a C++ binding and raises `set_group() takes no keyword arguments` on
    set_group(name, slot=...)."""
    try:
        from itasca import ballarray as ba
        ba.set_group(np.asarray(mask, dtype=bool), name, slot)
        return
    except (ImportError, AttributeError, TypeError):
        pass
    for b, k in zip(it.ball.list(), mask):
        if k:
            b.set_group(name, slot)


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
    set_group_mask(kill, "outside", "trim")
    cmd("ball delete range group 'outside' slot 'trim'")


def group_balls():
    """Ball groups only -- no contact properties yet.

    Relaxation destroys and recreates most of the contact list (19,248
    contacts before, 6,173 after, on the smoke-test pack), so anything
    written onto contacts at this point is thrown away and the survivors
    silently fall back to the cmat default. Contacts are assigned in
    assign_contacts(), after the pack has stopped moving.
    """
    x, z = data_frame_xz()
    u = m.unit(x, z)

    for name, code in UNITS.items():
        set_group_mask(u == code, name, "unit")
    weak = m.weak(x, z)
    set_group_mask(weak, "weak", "fault")
    set_group_mask(~weak, "intact", "fault")
    conv = m.conveyor(x, z)
    set_group_mask(conv, "conveyor", "drive")
    set_group_mask(~conv, "free", "drive")
    back = m.backstop(x, z)
    set_group_mask(back, "backstop", "bc")
    set_group_mask(~back, "interior", "bc")
    print(f"[groups] {conv.sum():,} conveyor, {back.sum():,} backstop, "
          f"{weak.sum():,} in seeded faults", flush=True)


def hold_boundaries():
    """The model has no walls. Its boundaries are the periodic y faces,
    a free toe and free surface, and two layers of fixed particles: the
    conveyor at the base and the backstop at the landward end.

    This has to happen BEFORE the pack is cycled. Without it nothing
    holds the wedge up at all and the whole thing falls out of the
    domain, which `condition destroy` then deletes without a word -- the
    symptom is an export of zero particles.
    """
    cmd("ball fix velocity spin range group 'conveyor' slot 'drive'")
    cmd("ball fix velocity spin range group 'backstop' slot 'bc'")


def relax():
    """Push out the overlaps `ball distribute` left behind, unbonded."""
    cmd("model clean")
    n0 = census("before relaxation")
    if RELAX_CYCLES:
        # `model cycle i calm i2` -- calm every i2 steps. PFC 6.0 wants
        # the interval: a bare `model cycle 1 calm` is a syntax error.
        cmd(f"model cycle {RELAX_CYCLES} calm {RELAX_CALM}")
        n1 = census("after relaxation")
        if n1 < n0:
            print(f"[relax] WARNING: lost {n0 - n1:,} balls "
                  f"({100 * (n0 - n1) / n0:.1f}%) out of the domain", flush=True)


def in_group(name, slot):
    """How many balls are in a group, asked of PFC rather than of the
    classification array. Relaxation can delete balls, so an array
    computed before it no longer lines up with the ball list; the group
    is what the `range` clauses below actually match on."""
    try:
        from itasca import ballarray as ba
        return int(np.asarray(ba.in_group(name, slot)).sum())
    except (ImportError, AttributeError, TypeError):
        return sum(1 for b in it.ball.list() if b.in_group(name, slot))


def assign_densities():
    """Per-unit ball density.

    Separate from assign_contacts() and called BEFORE the first cycle,
    for two reasons. A ball attribute survives relaxation, where a
    contact property does not. And a ball whose density was never set has
    zero inertial mass, which PFC refuses to cycle: `*** Ball 5367 has
    zero inertial mass.`
    """
    for name, p in PROPERTIES.items():
        n = in_group(name, "unit")
        if n == 0:
            continue
        cmd(f"ball attribute density {p['density']:.1f} "
            f"range group '{name}' slot 'unit'")
    missing = it.ball.count() - sum(in_group(k, "unit") for k in PROPERTIES)
    if missing:
        raise RuntimeError(
            f"{missing} balls are in no unit group and so have no density. "
            f"Every ball inside the envelope must get one, or PFC stops at "
            f"the first cycle with 'zero inertial mass'.")


def assign_contacts():
    # `contact property ... range` only reaches contacts that already
    # exist, and `ball distribute` creates balls, not contacts. `model
    # clean` runs contact detection without taking a timestep, so the
    # per-unit properties below land on a populated contact list.
    cmd("model clean")
    census("before contact properties")

    # A contact is inside `range group X slot 'unit'` if EITHER of its
    # two balls is in X, so a contact spanning a unit boundary matches
    # both units and keeps whichever was applied LAST. On this pack the
    # per-unit applications sum to 38,553 over 32,283 contacts, so about
    # 6,300 of them are interface contacts decided by application order.
    # Dict order is not a modelling decision, so apply strongest first
    # and weakest last: an interface then takes the weaker unit's
    # properties. That is the conservative choice, and it stops the
    # décollement -- the weakest unit, and the one the whole model turns
    # on -- from being overwritten by the underthrust section under it.
    for name in sorted(PROPERTIES, key=lambda k: -PROPERTIES[k]["cohesion"]):
        p = PROPERTIES[name]
        n = in_group(name, "unit")
        if n == 0:
            continue
        fric = BASAL_FRICTION if name == "decollement" else p["friction"]
        # intact rock, then the same rock inside a seeded fault. Applied
        # as two explicit values rather than a multiplier, because not
        # every PFC build accepts `multiply` on contact property.
        for tag, f in (("intact", 1.0), ("weak", WEAK_FACTOR)):
            rng = f"range group '{name}' slot 'unit' group '{tag}' slot 'fault'"
            # emod, kratio, pb_emod and pb_kratio are all READ-ONLY
            # properties of linearpbond -- PFC answers `Property
            # pb_kratio ... is read only!` and aborts. The stiffnesses
            # are set through the deformability methods, which is what
            # they are for; only the strengths and friction are
            # properties.
            cmd(f"contact method deformability "
                f"emod {p['E']:.6g} kratio 1.5 {rng}")
            cmd(f"contact method pb_deformability "
                f"emod {p['E']:.6g} kratio 1.5 {rng}")
            cmd(f"contact property "
                f"pb_ten {p['tensile'] * f:.6g} pb_coh {p['cohesion'] * f:.6g} "
                f"fric {fric:.3f} {rng}")
        print(f"[units] {name:<12} {n:7,}  fric {fric:.2f}  E {p['E']:.3g}", flush=True)
    print(f"[units] seeded faults bonded at {WEAK_FACTOR:.2f} of host strength",
          flush=True)


def settle():
    cmd("model calm")
    cmd("contact method bond gap 0.0")
    # `model solve ratio` is rejected by PFC 6.0 -- it answers "Bad
    # conversion of parameter number 3 (ratio)" and lists what it will
    # take: ratio-average, ratio-local, ratio-maximum. ratio-average is
    # the usual equilibrium criterion, the mean unbalanced force ratio.
    cmd("model solve ratio-average 1e-5"
        + (f" cycles {SOLVE_CYCLES}" if SOLVE_CYCLES else ""))
    census("after settling")
    cmd("ball attribute displacement 0 0 0")           # zero the datum
    print("[settle] equilibrated under tilted gravity", flush=True)


def drive():
    cmd("ball fix velocity range group 'conveyor' slot 'drive'")
    cmd(f"ball attribute velocity-x {PLATE_VELOCITY:.4g} "
        f"range group 'conveyor' slot 'drive'")
    steps = int(SHORTENING / PLATE_VELOCITY)
    print(f"[drive] {SHORTENING:.0f} m of convergence in {steps} steps", flush=True)
    cmd(f"model cycle {steps}")
    census("after convergence")


if __name__ == "__main__":
    build()
    group_balls()
    assign_densities()
    hold_boundaries()
    relax()
    assign_contacts()
    settle()
    pfc_export.export(STEM, reference=True)
    drive()
    pfc_export.export(STEM)
    print(f"[done] exported to {STEM}_*.txt", flush=True)
