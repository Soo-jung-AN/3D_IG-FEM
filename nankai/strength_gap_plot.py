"""Heatmaps of the strength x bonding-gap grid written by
nankai/strength_gap_sweep.py. Run it with --plot."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CSV, OUT = sys.argv[1], sys.argv[2]

# ink and chrome tokens
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
# sequential: one hue, light -> dark (blue ramp 100 -> 700)
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# a second sequential context on the same figure takes the next categorical
# hue (orange) as its own one-hue ramp, so the two panels are not read as
# the same scale. Stepped light -> dark off the documented orange anchor.
SEQ2 = LinearSegmentedColormap.from_list("seq_orange", [
    "#fdeae0", "#fac9b0", "#f6a980", "#f08a57", "#eb6834", "#bd4e1f", "#8a3612"])
# diverging: blue <-> red with a neutral gray midpoint
DIV = LinearSegmentedColormap.from_list("div_blue_red", [
    "#0d366b", "#256abf", "#86b6ef", "#cde2fb", "#f0efec",
    "#f3c4c4", "#e34948", "#d03b3b", "#8f2626"])

d = np.genfromtxt(CSV, delimiter=",", names=True)
S = np.unique(d["strength"]); G = np.unique(d["gap_frac"])
def grid(col):
    out = np.full((len(S), len(G)), np.nan)
    for r in np.atleast_1d(d):
        out[np.where(S == r["strength"])[0][0],
            np.where(G == r["gap_frac"])[0][0]] = r[col]
    return out

tot = grid("n_contacts")
bonded = 100.0 * grid("bonded_after") / tot
tension = 100.0 * grid("tension") / tot
err = grid("alpha0") - grid("alpha_target")

panels = [
    ("a  bonds surviving gravity", bonded, SEQ, None, "%", "%.0f"),
    ("b  contacts broken in TENSION", tension, SEQ2, None, "%", "%.0f"),
    ("c  alpha_0 error against the section", err, DIV,
     TwoSlopeNorm(vmin=min(-0.05, np.nanmin(err)), vcenter=0.0,
                  vmax=max(0.05, np.nanmax(err))), "°", "%+.2f"),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
fig.patch.set_facecolor(SURFACE)
for ax, (title, Z, cmap, norm, unit, fmt) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    im = ax.imshow(Z, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    lo, hi = np.nanmin(Z), np.nanmax(Z)
    for i in range(len(S)):
        for j in range(len(G)):
            v = Z[i, j]
            if np.isnan(v):
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax.text(j, i, fmt % v, ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if lum < 0.5 else INK)
    ax.set_xticks(range(len(G))); ax.set_xticklabels(["%.2f" % g for g in G])
    ax.set_yticks(range(len(S))); ax.set_yticklabels(["×%g" % s for s in S])
    ax.set_xlabel("bonding gap  (x r_min)", color=INK2, fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel("pb_ten / pb_coh multiplier", color=INK2, fontsize=10)
    ax.set_title(title + "   (" + unit + ")", fontsize=11, color=INK, pad=16)
    if "alpha_0" in title:
        ax.text(0.5, 1.015, "0 = the section;  |err| <= 0.05 is the "
                "calibration gate", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xticks(np.arange(-.5, len(G), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(S), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

fig.suptitle("bond strength x bonding gap, 6,627-ball pack -- every point stopped "
             "at the same 3,000 cycles, so none of them is an equilibrium", fontsize=11.5, color=INK, y=1.005)
fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("saved", OUT)
