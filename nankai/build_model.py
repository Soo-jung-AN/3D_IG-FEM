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

THE OPEN QUESTION is now a property problem, not a procedure one.

Three procedural faults have been found and fixed, and the numbers are
from the same 6,627-ball smoke pack throughout:

  - `contact method bond gap 0.0` bonded almost nothing, because
    relaxation exists to remove the overlaps it needs, and because a
    contact at a positive gap does not exist at all unless the CMAT
    proximity keeps it. Bonding across 0.2 r_min with a matching
    proximity bonds 63% of contacts instead.
  - the fixed-particle boundaries were 0.75 of a particle diameter thick
    at smoke resolution: a sieve, not a wall. The pack poured through the
    base and 92.5% of it was lost. Flooring them at three diameters
    brought that to 7.1%.
  - gravity was applied in one step to a pack bonded weightless.

What none of that fixed is alpha_0, and the gravity ramp says why. The
bonded skeleton loses 63% -> 39% of its bonds at ONE TENTH of gravity,
and ends at 16% whether gravity is ramped in ten steps or applied in
one. strength_check() prints the reason before the run starts: the wedge
is 7,900 m thick, so rho*g*H at its base is 209 MPa, and the STRONGEST
unit bonds at 10 MPa -- 4.8% of that. Every other unit is between 0.02%
and 1.4%. A skeleton two orders of magnitude weaker than the weight it
carries cannot stand, and no ordering of the commands will change it.

So the next decision is one about the model, not the script: either the
bond strengths in model_spec.PROPERTIES go up by orders of magnitude, or
the wedge is accepted as frictional and the bonded fraction stops being
treated as a measure of anything. Until alpha_0 agrees across friction
values to 0.05 deg, friction_sweep.report() refuses to name a calibrated
mu_b, and it is right to.
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
from nankai import taper                              # noqa: E402
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
# The settling solve. ratio-average 1e-5 was never reached -- it stalled
# two and a half orders of magnitude above it -- so the target is 1e-4
# and the cycle count is capped, because an unreachable target with no
# cap is a run that never ends.
SOLVE_RATIO = float(os.environ.get("NANKAI_SOLVE_RATIO", "1e-4"))
SOLVE_CYCLES = int(os.environ.get("NANKAI_SOLVE_CYCLES", "100000"))

# THE BONDING GAP, and why the wedge would not stand up.
#
# `contact method bond gap 0.0` bonds a contact only where the particles
# already overlap. Relaxation exists precisely to remove those overlaps,
# so by the time it is asked, almost nothing qualifies: the pack is left
# effectively cohesionless and spreads like sand.
#
# Worse, a contact at a positive gap does not EXIST unless the CMAT was
# told to keep it. From the linear parallel bond model manual: "One can
# ensure the existence of contacts between all pieces with a contact gap
# less than a specified bonding gap by specifying it with the proximity
# in the contact cmat default command." So the proximity below and the
# bonding gap are the same number, and both are set from the smallest
# particle radius rather than an absolute length, so the model stays
# scale free when R_MEAN is changed for a smoke test.
# Gravity is ramped rather than switched on. Applying it in one step to
# a pack that was bonded weightless is an impulse: on the smoke pack it
# broke 83% of the bonds in the first cycles. The ramp lets each
# increment of weight be carried before the next arrives.
GRAVITY_STEPS = int(os.environ.get("NANKAI_GRAVITY_STEPS", "10"))
GRAVITY_STEP_CYCLES = int(os.environ.get("NANKAI_GRAVITY_STEP_CYCLES", "500"))

# WHICH WEDGE. A real accretionary prism does not stand on the tensile
# strength of its own cement -- it stands in compression, on friction. The
# bonded variant was tried first and the numbers refused it: the strongest
# unit bonds at 4.8% of rho*g*H, and a strength x gap grid needed a x100
# multiplier (pb_ten 100 MPa, harder than granite) before the skeleton
# stopped shattering. "frictional" is therefore the default:
#
#   frictional  no parallel bonds anywhere. The pack carries its weight
#               the way sand does, through contact friction. A seeded
#               fault is a band of LOW FRICTION rather than weak cement,
#               which is what a fault zone is. An unbonded linearpbond
#               contact is identical to the linear model, so the contact
#               model itself does not change.
#   bonded      the earlier behaviour, kept so the two can be compared.
WEDGE = os.environ.get("NANKAI_WEDGE", "frictional")
# friction inside a seeded fault, as a fraction of the host unit's. The
# strength equivalent of WEAK_FACTOR, for a wedge held by friction.
WEAK_FRICTION = float(os.environ.get("NANKAI_WEAK_FRICTION", "0.5"))

# Where to drop a per-stage snapshot of the pack, for
# nankai/stage_figure.py. Empty means do not write any.
SNAPSHOT_DIR = os.environ.get("NANKAI_SNAPSHOTS", "")

R_MIN = R_MEAN / R_RATIO ** 0.5
BOND_GAP_FRAC = float(os.environ.get("NANKAI_BOND_GAP_FRAC", "0.2"))
BOND_GAP = BOND_GAP_FRAC * R_MIN
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
B_BETA = m.beta


def cmd(s):
    it.command(s)


_SNAP_N = [0]


def snapshot(tag, unit_codes=None):
    """Write the pack as it stands, so the run can be looked at rather
    than only summarised. ids are stored with it: particles are deleted
    as the run goes on, so a stage is only comparable to another one
    through the ids they share."""
    if not SNAPSHOT_DIR:
        return
    if not os.path.isdir(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)
    from itasca import ballarray as ba
    _SNAP_N[0] += 1
    out = os.path.join(SNAPSHOT_DIR, "%02d_%s.npz" % (_SNAP_N[0], tag))
    kw = dict(pos=np.asarray(ba.pos(), dtype=float),
              rad=np.asarray(ba.radius(), dtype=float),
              ids=np.asarray(ba.ids()).astype(np.int64),
              beta=B_BETA, tag=tag)
    if unit_codes is not None:
        kw["unit"] = np.asarray(unit_codes)
    np.savez(out, **kw)
    print(f"[snap] {tag:<24} -> {os.path.basename(out)}", flush=True)


def census(stage):
    """Balls and contacts at a stage. The domain is `condition destroy`,
    so a ball that leaves is deleted without comment -- a count that
    falls between two stages is the only warning you get, and it breaks
    the export invariant that the reference and deformed states hold the
    same particles in the same order."""
    n = it.ball.count()
    print(f"[census] {stage:<22} {n:>8,} balls  "
          f"{count_contacts():>9,} contacts "
          f"({count_contacts(False):,} active)", flush=True)
    return n


def build():
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
    # NO GRAVITY YET. The pack is relaxed weightless: `ball distribute`
    # leaves overlaps, and letting those overlaps unload under gravity at
    # the same time turns the relaxation into a collapse. Gravity is
    # switched on in apply_gravity(), after the pack is bonded.
    cmd("model gravity 0 0 0")
    cmd("model random 10101")
    # proximity exists so a contact at a positive gap survives to be
    # bonded. A frictional wedge bonds nothing, so it only needs the
    # contacts that actually touch.
    prox = BOND_GAP if WEDGE == "bonded" else 0.0
    cmd(f"contact cmat default model linearpbond "
        f"method deformability emod 1e9 kratio 1.5 "
        f"proximity {prox:.4f}")

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
    d = 2.0 * R_MEAN
    print(f"[groups] {conv.sum():,} conveyor ({m.conveyor_thickness():.0f} m "
          f"= {m.conveyor_thickness() / d:.1f} particle diameters), "
          f"{back.sum():,} backstop ({m.backstop_width():.0f} m "
          f"= {m.backstop_width() / d:.1f} diameters), "
          f"{weak.sum():,} in seeded faults", flush=True)
    if min(m.conveyor_thickness(), m.backstop_width()) < 2.0 * d:
        print("[groups] WARNING: a boundary thinner than two particle "
              "diameters is a sieve, not a wall -- the pack pours through it.",
              flush=True)
    snapshot("built", u)
    return u


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
    """Push out the overlaps `ball distribute` left behind.

    Weightless and unbonded: gravity is still 0 0 0 from build(), so the
    only thing driving the pack is its own overlap, and it has nothing to
    collapse under while it unloads.
    """
    cmd("model clean")
    n0 = census("before relaxation")
    if RELAX_CYCLES:
        # `model cycle i calm i2` -- calm every i2 steps. PFC 6.0 wants
        # the interval: a bare `model cycle 1 calm` is a syntax error.
        cmd(f"model cycle {RELAX_CYCLES} calm {RELAX_CALM}")
        n1 = census("after relaxation")
        snapshot("relaxed")
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


def strength_check():
    """Compare the bond strengths against the weight they have to carry.

    A bonded pack stands up only if its bonds can carry its own weight.
    The wedge is kilometres thick, so the stress at its base is rho*g*H,
    and that is the number the bond strengths have to be read against --
    not against each other. This is printed before the run commits to
    anything, because if the margin is negative no amount of procedure
    will keep the skeleton intact: a gravity ramp shows it breaking at
    10% of g just as a single step shows it breaking at 100%.
    """
    x = np.linspace(m.x0, m.x1, 400)
    thickness = (m._d("decollement", x) - m._d("seafloor", x)).max() * KM
    rho = max(p["density"] for p in PROPERTIES.values())
    sigma = rho * 9.81 * thickness

    if WEDGE != "bonded":
        print(f"[strength] frictional wedge: rho*g*H = {sigma / 1e6:.0f} MPa "
              f"at the base is carried by friction, not by cement, so the "
              f"bond strengths below are inert except inside seeded faults.",
              flush=True)
        return sigma
    strongest = max(PROPERTIES.items(), key=lambda kv: kv[1]["tensile"])
    print(f"[strength] wedge up to {thickness:,.0f} m thick: rho*g*H = "
          f"{sigma / 1e6:.0f} MPa at its base", flush=True)
    for name, p in sorted(PROPERTIES.items(), key=lambda kv: -kv[1]["tensile"]):
        print(f"[strength]   {name:<12} pb_ten {p['tensile'] / 1e6:7.2f} MPa "
              f"= {p['tensile'] / sigma:8.4f} x the basal stress", flush=True)
    if strongest[1]["tensile"] < sigma:
        print(f"[strength] WARNING: even {strongest[0]}, the strongest unit, "
              f"bonds at {strongest[1]['tensile'] / sigma:.4f} of the basal "
              f"stress. The bonded skeleton CANNOT carry the self weight and "
              f"will shatter as gravity comes on, whether it is ramped or "
              f"not. Raise the strengths in model_spec.PROPERTIES, or accept "
              f"a frictional wedge and stop reporting a bonded fraction.",
              flush=True)
    return sigma


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
        for tag in ("intact", "weak"):
            # In a bonded wedge a seeded fault is weaker CEMENT; in a
            # frictional one it is a band of lower FRICTION. Only one of
            # the two is the thing holding the wedge up, so only one of
            # them should be weakened.
            f = 1.0 if (tag == "intact" or WEDGE != "bonded") else WEAK_FACTOR
            mu = fric * (WEAK_FRICTION if (tag == "weak" and
                                           WEDGE == "frictional") else 1.0)
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
                f"fric {mu:.3f} {rng}")
        print(f"[units] {name:<12} {n:7,}  fric {fric:.2f}  E {p['E']:.3g}", flush=True)
    print(f"[units] seeded faults bonded at {WEAK_FACTOR:.2f} of host strength",
          flush=True)


def bond():
    """Install the parallel bonds, across a gap rather than on overlap.

    `gap 0.0` bonds only what already overlaps, and relaxation has just
    removed the overlaps, so it leaves a cohesionless pack. The gap is a
    fraction of the smallest radius and matches the CMAT proximity, so
    the contacts it needs are there to be bonded.
    """
    cmd("model calm")
    if WEDGE != "bonded":
        print("[bond] frictional wedge: no parallel bonds installed. The "
              "pack carries its weight through contact friction, and a "
              "seeded fault is a low-friction band rather than weak "
              "cement.", flush=True)
        return 0
    cmd(f"contact method bond gap {BOND_GAP:.4f}")
    n, bonded = bond_census()
    print(f"[bond] gap {BOND_GAP:.1f} m = {BOND_GAP_FRAC:.2f} x r_min "
          f"({R_MIN:.1f} m): {bonded:,} of {n:,} contacts bonded "
          f"({100.0 * bonded / n if n else 0.0:.1f}%)", flush=True)
    return bonded


def contacts(all_contacts=True):
    """Ball-ball contact iterator.

    Two things the signature will not forgive. The FIRST argument is the
    process name, not the contact type: passing "ball-ball" there raises
    `ValueError: Unknown process name`. And only that first argument is
    positional -- `type` and `all` must be passed by keyword, or PFC
    answers `TypeError: function takes at most 1 argument (3 given)`.

    `all` includes virtual contacts, those inside the CMAT proximity but
    not yet touching. They matter here: they are exactly the contacts the
    bonding gap is meant to catch.
    """
    return it.contact.list("mechanical", type=it.BallBallContact,
                           all=all_contacts)


def count_contacts(all_contacts=True):
    return it.contact.count("mechanical", type=it.BallBallContact,
                            all=all_contacts)


def bond_census():
    """(contacts, bonded). pb_state is 0 unbonded, 1 broke in tension,
    2 broke in shear, 3 bonded, so only 3 counts as a live bond."""
    n = bonded = 0
    for c in contacts():
        n += 1
        try:
            bonded += (c.prop("pb_state") == 3)
        except Exception:
            pass
    return n, bonded


def apply_gravity():
    """Bring gravity up, now that there is a bonded skeleton to carry it.

    Until this point the model has been weightless. Full gravity in one
    command is an impulse on a pack that has never felt any, so it is
    ramped and the bond count is reported at each step: a ramp that still
    shatters the skeleton says the bonds are too weak for the self
    weight, which is a property problem, not a procedure one.
    """
    gx, gy, gz = m.gravity()
    steps = max(1, GRAVITY_STEPS)
    for i in range(1, steps + 1):
        f = float(i) / steps
        cmd(f"model gravity {gx * f:.6f} {gy * f:.6f} {gz * f:.6f}")
        if GRAVITY_STEP_CYCLES:
            cmd(f"model cycle {GRAVITY_STEP_CYCLES}")
        n, bonded = bond_census()
        print(f"[gravity] {100.0 * f:5.1f}% of g   {it.ball.count():>7,} balls   "
              f"{bonded:>7,} of {n:,} contacts bonded "
              f"({100.0 * bonded / n if n else 0.0:.1f}%)", flush=True)
    print(f"[gravity] full g = ({gx:+.3f}, {gy:.3f}, {gz:.3f}) m/s2, tilted "
          f"{m.beta:.2f} deg with the decollement", flush=True)
    snapshot("gravity_on")


def settle():
    # `model solve ratio` is rejected by PFC 6.0 -- it answers "Bad
    # conversion of parameter number 3 (ratio)" and lists what it will
    # take: ratio-average, ratio-local, ratio-maximum. ratio-average is
    # the usual equilibrium criterion, the mean unbalanced force ratio.
    before = it.cycle()
    cmd(f"model solve ratio-average {SOLVE_RATIO:.3g}"
        + (f" cycles {SOLVE_CYCLES}" if SOLVE_CYCLES else ""))
    spent = it.cycle() - before
    census("after settling")
    snapshot("settled")
    cmd("ball attribute displacement 0 0 0")           # zero the datum
    return spent


def report_settling(n_built, bonded_at_bond, cycles_spent):
    """What the settled wedge actually is, before any convergence.

    Three numbers, because between them they say whether the run is worth
    continuing: how much of the bonded skeleton survived gravity, how
    much of the pack was lost, and whether the surface still matches the
    section it was built from.
    """
    n_now = it.ball.count()
    lost = n_built - n_now
    n_contacts = count_contacts()
    bonded, broke = 0, 0
    if WEDGE == "bonded":
        n_contacts, bonded = bond_census()
        broke = bonded_at_bond - bonded

    a0, _, _ = taper.measure(ball_positions(), m.beta)
    target = taper.target_alpha()

    print("\n[settled] --------------------------------------------------",
          flush=True)
    if WEDGE == "bonded":
        print(f"[settled] bonded contacts   {bonded:,} of {n_contacts:,} "
              f"({100.0 * bonded / n_contacts if n_contacts else 0.0:.1f}%), "
              f"{broke:,} broke under gravity", flush=True)
    else:
        print(f"[settled] contacts          {n_contacts:,} "
              f"(frictional wedge -- no bonds, so no bonded fraction to "
              f"report)", flush=True)
    print(f"[settled] particles lost    {lost:,} of {n_built:,} "
          f"({100.0 * lost / n_built if n_built else 0.0:.1f}%)", flush=True)
    print(f"[settled] alpha_0           {a0:.3f} deg  "
          f"(section: {target:.3f} deg, error {a0 - target:+.3f})", flush=True)

    # Did the solve converge, or did it merely stop? This matters more
    # than it looks. On the smoke pack 300 cycles leave alpha_0 at 2.451
    # deg, 0.049 off the section, and lose no particles; 30,000 cycles
    # leave it at 6.973 deg and lose 8.3%. A wedge whose answer depends
    # on the cycle cap is creeping, not equilibrating, and the cap is
    # then a tuning knob that flatters the result.
    hit_cap = SOLVE_CYCLES and cycles_spent >= SOLVE_CYCLES
    print(f"[settled] settling solve    {cycles_spent:,} cycles"
          f"{' -- STOPPED AT THE CAP' if hit_cap else ' -- reached the ratio'}",
          flush=True)
    if hit_cap:
        print(f"[settled] WARNING: the solve hit its cycle cap instead of "
              f"reaching ratio-average {SOLVE_RATIO:.0e}, so this is not an "
              f"equilibrium and alpha_0 above is only where the wedge had got "
              f"to. Re-run with a different NANKAI_SOLVE_CYCLES: if alpha_0 "
              f"moves, the wedge is creeping and the cap is choosing the "
              f"answer.", flush=True)

    # alpha_0 is measured BEFORE any convergence, from the geometry the
    # model was built to. It should therefore be the section's alpha, and
    # it should not depend on basal friction at all. If it does, the
    # settling stage is what the friction sweep is measuring.
    if abs(a0 - target) > 0.25:
        print(f"[settled] WARNING: alpha_0 is {abs(a0 - target):.2f} deg off "
              f"the section before any convergence. The wedge is not holding "
              f"the geometry it was built to, so a friction sweep measures "
              f"settling, not friction.", flush=True)
    print("[settled] --------------------------------------------------\n",
          flush=True)
    return dict(alpha_0=a0, target=target, lost=lost, n_built=n_built,
                bonded=bonded, contacts=n_contacts)


def drive():
    cmd("ball fix velocity range group 'conveyor' slot 'drive'")
    cmd(f"ball attribute velocity-x {PLATE_VELOCITY:.4g} "
        f"range group 'conveyor' slot 'drive'")
    steps = int(SHORTENING / PLATE_VELOCITY)
    print(f"[drive] {SHORTENING:.0f} m of convergence in {steps} steps", flush=True)
    cmd(f"model cycle {steps}")
    census("after convergence")
    snapshot("converged")


if __name__ == "__main__":
    build()
    strength_check()
    n_built = it.ball.count()
    group_balls()
    assign_densities()
    hold_boundaries()
    relax()                  # weightless: overlaps out before anything else
    assign_contacts()
    bonded_at_bond = bond()  # across a gap, not on overlap
    apply_gravity()          # only now does the model have weight
    cycles_spent = settle()
    report_settling(n_built, bonded_at_bond, cycles_spent)
    pfc_export.export(STEM, reference=True)
    drive()
    pfc_export.export(STEM)
    print(f"[done] exported to {STEM}_*.txt", flush=True)
