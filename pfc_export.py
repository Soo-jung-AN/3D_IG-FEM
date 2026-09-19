"""Export a PFC3D model state into the text format this repository reads.

RUNS INSIDE PFC. `itasca` is only importable from the Python interpreter
embedded in PFC, so this is called by a PFC session, not by a standalone
python. pfc_pipeline.py does that for you.

It writes the five files that txt_strikeslip/ holds, so everything
downstream -- make_mesh.py, vp_from_strain.py, strain_analysis.py,
seismic_section.py -- works on the output with no conversion:

    <stem>_init_pos.txt     undeformed centres, (x, y, z) in m
    <stem>_pos.txt          deformed centres, same particle ordering
    <stem>_rad.txt          radii, m
    <stem>_density.txt      per-particle density, kg/m3
    <stem>_contactF.txt     per-particle resultant contact force, N

Call it twice in a run: once on the equilibrated reference state with
`reference=True`, which writes only <stem>_init_pos.txt, and again after
deformation, which writes the rest.

NOTE ON THE ITASCA API. The accessors below differ between PFC 6, 7 and
8. Each getter tries the vectorised `ballarray` form first, then a
per-ball loop, and raises with the names it tried if neither works.

VERIFIED on PFC 6.00 Release 008: ballarray.pos() -> (n, 3),
.radius() -> (n,), .density() -> (n,), .force_contact() -> (n, 3) and
.ids() -> (n,) int64 all exist and return those shapes, so the
vectorised path is the one that runs. Ball.pos/.radius/.density/
.force_contact/.id exist too, so the fallback is good as well. If a
later PFC disagrees, fix the one function and everything else still
holds; `help(itasca.ballarray)` inside PFC lists what your build has.

PFC 6.0 embeds numpy 1.13, which is old enough to matter: no
np.trapezoid, and np.intersect1d has no return_indices. Anything in
this file has to run there as well as on a current numpy.
"""
import os

import numpy as np

import itasca as it                                    # noqa: F401  (PFC only)

try:
    from itasca import ballarray as ba
except ImportError:                                     # older PFC
    ba = None


def _vector(name, array_fn, ball_fn, n=None):
    """Try the vectorised accessor, fall back to a per-ball loop."""
    if ba is not None:
        fn = getattr(ba, array_fn, None)
        if fn is not None:
            return np.asarray(fn(), dtype=float)
    balls = list(it.ball.list())
    for attr in ball_fn:
        if hasattr(balls[0], attr):
            return np.array([getattr(b, attr)() for b in balls], dtype=float)
    raise AttributeError(
        f"cannot read {name}: tried ballarray.{array_fn} and ball."
        f"{{{', '.join(ball_fn)}}}. Run `help(itasca.ballarray)` and "
        f"`help(itasca.ball)` inside PFC and fix _vector() for your version.")


def positions():
    return _vector("positions", "pos", ("pos",))


def radii():
    return _vector("radii", "radius", ("radius",))


def densities():
    return _vector("densities", "density", ("density",))


def contact_forces():
    return _vector("contact forces", "force_contact",
                   ("force_contact", "contact_force"))


def ball_ids():
    """Particle ids, so the two exports can be checked for a consistent
    ordering rather than assumed to have one."""
    try:
        return _vector("ids", "ids", ("id",)).astype(np.int64)
    except AttributeError:
        return None


def _positions_of(values, ids):
    """Row index of each id in `values` within the array `ids`.

    np.intersect1d's return_indices argument needs numpy >= 1.15 and PFC
    6.0 embeds numpy 1.13, so the lookup is done by hand.
    """
    order = np.argsort(ids)
    return order[np.searchsorted(ids[order], values)]


def _match_by_id(stem, pos, ids):
    """Reduce the reference and deformed states to the particles they
    share, ordered the same way in both.

    A wedge with a free toe and a free surface loses particles: they slide
    off the front, leave the domain, and `condition destroy` deletes them.
    The strain solvers need one fixed set of particles in one fixed order,
    so rather than failing the whole run over it, the two states are
    intersected on particle id and the reference file is rewritten to
    match. Returns the row index into the deformed arrays, or None when
    nothing needs dropping.
    """
    init_file = f"{stem}_init_pos.txt"
    ids_file = f"{stem}_ids_init.txt"
    if not (os.path.exists(init_file) and os.path.exists(ids_file)) or ids is None:
        return None

    pos0 = np.loadtxt(init_file)
    ids0 = np.loadtxt(ids_file).astype(np.int64)
    if len(ids0) == len(ids) and np.array_equal(ids0, ids):
        return None

    common = np.intersect1d(ids0, ids)
    if len(common) == 0:
        raise RuntimeError(
            f"the reference export and this one share no particle ids at "
            f"all ({len(ids0)} and {len(ids)} particles). Something "
            f"rebuilt the pack between the two exports.")

    lost = len(ids0) - len(common)
    print(f"[pfc_export] {lost} of {len(ids0)} particles "
          f"({100.0 * lost / len(ids0):.1f}%) left the model between the "
          f"reference state and this one; exporting the {len(common)} "
          f"they share, matched on particle id.")
    if lost > 0.1 * len(ids0):
        print(f"[pfc_export] WARNING: that is more than 10%. A wedge that "
              f"sheds this much is not in equilibrium -- check the settling "
              f"stage before trusting the strain.")

    np.savetxt(init_file, pos0[_positions_of(common, ids0)])
    np.savetxt(ids_file, common, fmt="%d")
    return _positions_of(common, ids)


def export(stem, reference=False):
    out_dir = os.path.dirname(stem)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pos = positions()
    ids = ball_ids()
    if ids is not None:
        np.savetxt(f"{stem}_ids{'_init' if reference else ''}.txt", ids, fmt="%d")

    if reference:
        np.savetxt(f"{stem}_init_pos.txt", pos)
        print(f"[pfc_export] reference state: {len(pos)} particles -> "
              f"{stem}_init_pos.txt")
        return

    rad, den, fc = radii(), densities(), contact_forces()
    keep = _match_by_id(stem, pos, ids)
    if keep is not None:
        pos, rad, den, fc = pos[keep], rad[keep], den[keep], fc[keep]
        np.savetxt(f"{stem}_ids.txt", ids[keep], fmt="%d")

    np.savetxt(f"{stem}_pos.txt", pos)
    np.savetxt(f"{stem}_rad.txt", rad)
    np.savetxt(f"{stem}_density.txt", den)
    np.savetxt(f"{stem}_contactF.txt", fc)
    print(f"[pfc_export] deformed state: {len(pos)} particles -> {stem}_*.txt")

    n0 = len(np.loadtxt(f"{stem}_init_pos.txt"))
    if n0 != len(pos):
        raise RuntimeError(
            f"particle count still differs after matching on id: reference "
            f"{n0}, deformed {len(pos)}. The strain solvers assume a fixed "
            f"set of particles in a fixed order. This means the run has no "
            f"usable particle ids -- check that ball_ids() works on your "
            f"PFC build, because without them the two states cannot be "
            f"put in correspondence at all.")
