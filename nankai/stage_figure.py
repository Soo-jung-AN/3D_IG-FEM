"""Draw what the pack actually does, stage by stage.

build_model.py writes a snapshot at each stage when NANKAI_SNAPSHOTS is
set. This turns them into a section: the model frame is rotated back into
the data frame, so what is drawn is comparable with the digitised
transect rather than with the tilted frame the DEM runs in.

    NANKAI_SNAPSHOTS=runs/snap_frictional \
    python3 pfc_pipeline.py nankai/build_model.py --stem runs/f01 --only pfc
    python3 nankai/stage_figure.py runs/snap_frictional

Two rows. The top is the pack coloured by unit, which says where the
material is; the bottom is displacement from the built state, which says
what moved. A wedge that is holding its shape has a blue top row that
still matches the section and a bottom row that is almost empty.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nankai import geometry as G
from nankai import taper
from nankai.model_spec import KM, UNIT_NAMES, Model

# chart chrome; one hue per unit, fixed order, never cycled
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
UNIT_COLOUR = {
    "kumano": "#eda100", "inner_prism": "#4a3aa7", "outer_prism": "#2a78d6",
    "decollement": "#e34948", "underthrust": "#1baf7a", "crust": "#52514e",
}
# sequential one-hue ramp for displacement magnitude
DISP_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]


def load(d):
    snaps = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        snaps.append(dict(tag=str(z["tag"]), pos=z["pos"], rad=z["rad"],
                          ids=z["ids"], beta=float(z["beta"]),
                          unit=z["unit"] if "unit" in z.files else None))
    if not snaps:
        raise SystemExit("no snapshots in %s -- set NANKAI_SNAPSHOTS and "
                         "re-run build_model.py" % d)
    return snaps


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    d = sys.argv[1] if len(sys.argv) > 1 else "runs/snap"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fig_stages.png")
    snaps = load(d)
    m = Model(beta=snaps[0]["beta"])
    base = snaps[0]
    unit_of = dict(zip(base["ids"].tolist(), base["unit"].tolist()))
    pos0 = dict(zip(base["ids"].tolist(), map(tuple, base["pos"])))
    cmap = LinearSegmentedColormap.from_list("disp", DISP_STEPS)

    # a common displacement scale, so the panels are comparable
    dmax = 0.0
    for s in snaps[1:]:
        p0 = np.array([pos0.get(i, (np.nan,)*3) for i in s["ids"].tolist()])
        dmax = max(dmax, np.nanpercentile(
            np.linalg.norm(s["pos"] - p0, axis=1), 98))
    dmax = max(dmax, 1.0)

    n = len(snaps)
    # 45 km across and ~10 km deep at VE 1.5 is a 3:1 panel, so the rows
    # are short and wide; a third, full-width row overlays the surfaces,
    # which is where alpha actually lives.
    fig = plt.figure(figsize=(4.7 * n, 9.0))
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.0, 1.25],
                          hspace=0.62, wspace=0.14)
    fig.patch.set_facecolor(SURFACE)
    sf = np.linspace(0, 45, 400)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(2)]
    ax_s = fig.add_subplot(gs[2, :])
    surf_colour = ["#0b0b0b", "#2a78d6", "#1baf7a", "#eb6834", "#e34948"]

    for k, s in enumerate(snaps):
        x, z = m.to_data(s["pos"][:, 0], s["pos"][:, 2])
        x, depth = x / KM, -z / KM
        u = np.array([unit_of.get(i, -1) for i in s["ids"].tolist()])
        a, _, _ = taper.measure(s["pos"], m.beta)

        ax = axes[0][k]
        for code, name in UNIT_NAMES.items():
            q = u == code
            if q.any():
                ax.scatter(x[q], depth[q], s=1.1, lw=0,
                           c=UNIT_COLOUR[name], label=name)
        ax.plot(sf, G.depth_at("seafloor", sf), color=INK, lw=1.3, ls="--")
        ax.plot(sf, G.depth_at("decollement", sf), color=INK, lw=1.0, ls=":")
        ax.set_title("%d  %s" % (k + 1, s["tag"]), fontsize=11.5, color=INK,
                     pad=16)
        ax.text(0.5, 1.02, "alpha_0 %.3f deg   err %+.3f"
                % (a, a - taper.target_alpha()), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5,
                color="#0b0b0b" if abs(a - taper.target_alpha()) < 0.3 else "#d03b3b")

        ax2 = axes[1][k]
        p0 = np.array([pos0.get(i, (np.nan,) * 3) for i in s["ids"].tolist()])
        disp = np.linalg.norm(s["pos"] - p0, axis=1)
        sc = ax2.scatter(x, depth, s=1.1, lw=0, c=disp, cmap=cmap,
                         norm=Normalize(0, dmax))
        ax2.plot(sf, G.depth_at("seafloor", sf), color=INK, lw=1.3, ls="--")
        p90 = np.nanpercentile(disp, 90) if np.isfinite(disp).any() else 0.0
        ax2.set_title("displacement from stage 1   90th pct %.0f m" % p90,
                      fontsize=9.5, color=INK2, pad=8)

        # the surface itself, all stages on one axis
        xc, top = taper.surface_profile(
            *m.to_data(s["pos"][:, 0], s["pos"][:, 2]))
        ax_s.plot(xc, top, color=surf_colour[k % len(surf_colour)], lw=1.9,
                  label="%d %s  (alpha_0 %.2f)" % (k + 1, s["tag"], a))

        for a_ in (ax, ax2):
            a_.set_facecolor(SURFACE)
            a_.set_xlim(45, 0); a_.set_ylim(9.4, -0.6)
            a_.set_aspect(G.VE)
            a_.set_xlabel("distance from the trench (km)", color=INK2, fontsize=9)
            a_.tick_params(colors=MUTED, labelsize=8.5, length=2)
            for sp in a_.spines.values():
                sp.set_color(GRID)
            if k == 0:
                a_.set_ylabel("depth (km)", color=INK2, fontsize=9)

    ax_s.plot(sf, G.depth_at("seafloor", sf), color=MUTED, lw=2.4, ls="--",
              label="digitised sea floor")
    ax_s.axvspan(2.0, 29.0, color="#f0efec", zorder=0)
    ax_s.text(28.4, 0.06, "alpha_0 is fitted over this window",
              fontsize=8.5, color=MUTED, va="top")
    ax_s.set_facecolor(SURFACE)
    ax_s.set_xlim(45, 0); ax_s.set_ylim(5.2, -0.3)
    ax_s.set_xlabel("distance from the trench (km)", color=INK2, fontsize=9.5)
    ax_s.set_ylabel("sea-floor depth (km)", color=INK2, fontsize=9.5)
    ax_s.set_title("the surface, every stage on one axis -- this is what "
                   "alpha_0 measures", fontsize=11, color=INK, pad=8)
    ax_s.tick_params(colors=MUTED, labelsize=9, length=2)
    ax_s.grid(color=GRID, lw=0.7)
    ax_s.set_axisbelow(True)
    for sp in ax_s.spines.values():
        sp.set_color(GRID)
    ax_s.legend(frameon=False, fontsize=9, ncol=3, loc="lower left",
                labelcolor=INK2)

    handles = [Line2D([], [], marker="o", ls="", ms=6, color=UNIT_COLOUR[v],
                      label=v) for v in UNIT_NAMES.values()]
    handles += [Line2D([], [], color=INK, lw=1.3, ls="--", label="sea floor (section)"),
                Line2D([], [], color=INK, lw=1.0, ls=":", label="decollement (section)")]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=8,
               loc="lower center", bbox_to_anchor=(0.5, 0.955),
               labelcolor=INK2)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.008, pad=0.008)
    cb.set_label("displacement (m)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)
    cb.outline.set_edgecolor(GRID)

    fig.savefig(out, dpi=145, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
