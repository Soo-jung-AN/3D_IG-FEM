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
8, and this file has not been run against a PFC installation -- it was
written from the outside. Each getter tries the vectorised `ballarray`
form first, then a per-ball loop, and raises with the names it tried if
neither works. If it raises on your version, fix the one function and
everything else still holds. `python -c "import itasca; help(itasca)"`
inside PFC lists what your build actually exposes.
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

    np.savetxt(f"{stem}_pos.txt", pos)
    np.savetxt(f"{stem}_rad.txt", radii())
    np.savetxt(f"{stem}_density.txt", densities())
    np.savetxt(f"{stem}_contactF.txt", contact_forces())
    print(f"[pfc_export] deformed state: {len(pos)} particles -> {stem}_*.txt")

    init = f"{stem}_init_pos.txt"
    if os.path.exists(init):
        n0 = len(np.loadtxt(init))
        if n0 != len(pos):
            raise RuntimeError(
                f"particle count changed between the reference export "
                f"({n0}) and this one ({len(pos)}). The strain solvers "
                f"assume a fixed set of particles in a fixed order; "
                f"export the reference AFTER the pack is final, and do "
                f"not create or delete balls during the run.")
