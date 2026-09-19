"""PFC3D -> strain -> Vp -> synthetic seismic, in one command.

Replaces the manual loop: run PFC in the GUI, export a point .vtk, open
ParaView, apply Delaunay3D, save a .vtu, then run the analysis by hand.

    python3 pfc_pipeline.py model.py --stem runs/ss01 --alpha 125 --zmax -100

Stages, each skippable so a failed run can be resumed:

  1. pfc      run `model.py` inside PFC in batch. That script builds and
              cycles the model and calls pfc_export.export() twice --
              once with reference=True on the equilibrated pack, once at
              the end. See pfc_export.py.
  2. mesh     make_mesh.py, Qhull rather than ParaView's Delaunay3D
              (which drops 14.5% of a DEM pack; see that module).
  3. strain   strain_analysis.py -- the full tensor, not just det(F) - 1.
  4. vp       vp_from_strain.py -- Vp, Vs, rho from the volumetric strain
              and the run's own per-particle density.
  5. seismic  seismic_section.py -- velocity sections and the
              1D-convolution synthetic section.

FINDING THE PFC EXECUTABLE. Stage 1 needs to know how to launch PFC in
batch, and that differs across PFC 6/7/8. Pass --pfc-exe, or set
PFC_EXE, or put it in pfc_pipeline.json next to this file:

    {"pfc_exe": "C:/Program Files/Itasca/PFC700/exe64/pfc3d700.exe",
     "batch_args": ["-c", "python-reset-state false", "call", "{script}"]}

`{script}` is substituted with the Python file to run. The default
guess below is only a guess -- run PFC's executable with no arguments
once, or check its documentation, and record what works in the json so
it is not guessed again. Everything downstream of stage 1 is tested;
stage 1 is the part that depends on your installation.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "pfc_pipeline.json")
DEFAULT_BATCH_ARGS = ["call", "{script}"]      # a guess; see the docstring
STAGES = ("pfc", "mesh", "strain", "vp", "seismic")


def load_config():
    if os.path.exists(CONFIG):
        with open(CONFIG) as f:
            return json.load(f)
    return {}


def run(cmd, label):
    print(f"\n=== {label} ===\n$ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {r.returncode}")
    print(f"--- {label} done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?",
                    help="the Python file PFC runs (stage 1). Omit with "
                         "--skip pfc to analyse an existing export.")
    ap.add_argument("--stem", required=True,
                    help="path prefix the exports share, e.g. runs/ss01. "
                         "pfc_export writes <stem>_init_pos.txt etc.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None, help="figure suffix; default: the stem's basename")
    ap.add_argument("--alpha", type=float, default=None,
                    help="max tetrahedron edge in m (default: 3x the median "
                         "nearest-neighbour spacing)")
    ap.add_argument("--q-min", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=None,
                    help="drop particles above this z, e.g. an unconfined cap")
    ap.add_argument("--freq", type=float, default=25.0)
    ap.add_argument("--dx", type=float, default=20.0)
    ap.add_argument("--pfc-exe", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    ap.add_argument("--only", nargs="*", default=None, choices=STAGES)
    a = ap.parse_args()

    cfg = load_config()
    wanted = [s for s in STAGES if s not in a.skip and (a.only is None or s in a.only)]
    print(f"stages: {', '.join(wanted) if wanted else '(none)'}")

    tag = a.tag or os.path.basename(a.stem)
    os.makedirs(a.results, exist_ok=True)
    init = f"{a.stem}_init_pos.txt"
    pos = f"{a.stem}_pos.txt"
    density = f"{a.stem}_density.txt"
    mesh = f"{a.stem}_mesh.vtu"
    strain_npz = f"{a.results}/strain_{tag}.npz"
    vp_npz = f"{a.results}/vp_{tag}.npz"
    py = sys.executable

    if "pfc" in wanted:
        if not a.model:
            raise SystemExit("stage 'pfc' needs a model script; pass one or --skip pfc")
        exe = a.pfc_exe or os.environ.get("PFC_EXE") or cfg.get("pfc_exe")
        if not exe:
            raise SystemExit(
                "PFC executable not known. Pass --pfc-exe, set PFC_EXE, or write\n"
                f"  {CONFIG}\n"
                '  {"pfc_exe": "...pfc3d700.exe", "batch_args": ["call", "{script}"]}\n'
                "See the module docstring.")
        args = cfg.get("batch_args", DEFAULT_BATCH_ARGS)
        run([exe] + [s.format(script=os.path.abspath(a.model)) for s in args],
            f"1. PFC: {a.model}")
        for f in (init, pos, density):
            if not os.path.exists(f):
                raise SystemExit(
                    f"PFC finished but {f} is missing. Check that your model "
                    f"script calls pfc_export.export('{a.stem}', reference=True) "
                    f"on the equilibrated pack and pfc_export.export('{a.stem}') "
                    f"at the end.")

    if "mesh" in wanted:
        cmd = [py, f"{HERE}/make_mesh.py", init, mesh, "--q-min", a.q_min]
        if a.alpha:
            cmd += ["--alpha-edge", a.alpha]
        run(cmd, "2. mesh (Qhull)")

    if "strain" in wanted:
        cmd = [py, f"{HERE}/strain_analysis.py", "--init", init, "--pos", pos,
               "--q-min", a.q_min, "--out", strain_npz]
        if a.alpha:
            cmd += ["--alpha", a.alpha]
        if a.zmax is not None:
            cmd += ["--zmax", a.zmax]
        run(cmd, "3. strain tensor")

    if "vp" in wanted:
        cmd = [py, f"{HERE}/vp_from_strain.py", "--init", init, "--pos", pos,
               "--density", density, "--q-min", a.q_min, "--out", vp_npz]
        if a.alpha:
            cmd += ["--alpha", a.alpha]
        if a.zmax is not None:
            cmd += ["--zmax", a.zmax]
        run(cmd, "4. Vp from strain")

    if "seismic" in wanted:
        run([py, f"{HERE}/seismic_section.py", vp_npz, tag,
             "--dx", a.dx, "--freq", a.freq], "5. synthetic seismic")

    print(f"\nall done. figures: fig_seis_{tag}_*.png   arrays: {vp_npz}, {strain_npz}")


if __name__ == "__main__":
    main()
