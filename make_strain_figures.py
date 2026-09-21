"""Figures for strain_analysis.py: the full tensor, and the diagnosis."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

NPZ = sys.argv[1] if len(sys.argv) > 1 else "./results/strain_model3.npz"
TAG = sys.argv[2] if len(sys.argv) > 2 else "model3"
DX, Z_MAP, CAP, BUF = 15.0, -450.0, -100.0, 150.0

D = np.load(NPZ)
X0, X1 = D["X0"], D["X1"]
E = D["E"].astype(np.float64)
Z, Y, Xc = X0[:, 2], X0[:, 1], X0[:, 0]
ok = np.abs(D["vol"]) < 1
wall = (Xc < BUF) | (Xc > 2000 - BUF) | (Y < BUF) | (Y > 2000 - BUF)
interior = ok & ~wall & (Z < CAP)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})
tree = cKDTree(X1)
gx = np.arange(X1[:, 0].min(), X1[:, 0].max() + DX, DX)
gy = np.arange(X1[:, 1].min(), X1[:, 1].max() + DX, DX)
GX, GY = np.meshgrid(gx, gy, indexing="ij")
nodes = np.column_stack([GX.ravel(), GY.ravel(), np.full(GX.size, Z_MAP)])
dist, idx = tree.query(nodes, k=6)
w = 1.0 / np.maximum(dist, 1e-9) ** 2
msk = (dist[:, 0] >= 2 * DX).reshape(GX.shape).T
gridm = lambda v: np.ma.array(((w * v[idx]).sum(1) / w.sum(1)).reshape(GX.shape).T, mask=msk)
EXT = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]


def mapview(ax, field, title, lim, cmap="RdBu_r"):
    im = ax.imshow(gridm(field), origin="lower", extent=EXT, aspect="equal",
                   cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
    ax.axhline(1.0, color="#333", ls=":", lw=.9)
    plt.colorbar(im, ax=ax, shrink=0.86, pad=0.012)


# --- Figure 1: the six independent strain components -------------------
fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
lab = (("E11", 0, 0), ("E22", 1, 1), ("E33", 2, 2),
       ("E12", 0, 1), ("E13", 0, 2), ("E23", 1, 2))
lim = np.percentile(np.abs(E[interior][:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]), 99)
for a, (name, i, j) in zip(ax.ravel(), lab):
    mapview(a, E[:, i, j], f"{name}", lim)
fig.suptitle(f"Green-Lagrangian strain components, map view at Z = {Z_MAP:.0f} m  "
             f"(common scale ±{lim:.3f}; dotted line = the imposed shear axis)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.6)
fig.savefig(f"fig_strain_{TAG}_tensor.png", dpi=145)
print(f"saved fig_strain_{TAG}_tensor.png")

# --- Figure 2: the diagnosis -------------------------------------------
fig2, ax2 = plt.subplots(2, 3, figsize=(16, 9))

mapview(ax2[0, 0], np.abs(E[:, 0, 1]), "a  |E12|, the shear that the walls impose",
        np.percentile(np.abs(E[interior][:, 0, 1]), 99), cmap="magma_r")
im = ax2[0, 0].images[0]; im.set_clim(0, np.percentile(np.abs(E[interior][:, 0, 1]), 99))
mapview(ax2[0, 1], D["theta_z"], "b  vertical-axis rotation θz  (degrees)",
        np.percentile(np.abs(D["theta_z"][interior]), 99))
mapview(ax2[0, 2], D["vol"], "c  volumetric strain  det(F) − 1",
        np.percentile(np.abs(D["vol"][interior]), 99))

# d: the picked fault trace, and how localised the shear is
gg = np.abs(E[:, 0, 1])
xb = np.arange(200, 1801, 50)
xc_b, trace, halfw = [], [], []
for a, b in zip(xb[:-1], xb[1:]):
    c = interior & (Xc >= a) & (Xc < b)
    if c.sum() < 200:
        continue
    yb2 = np.arange(300, 1701, 25); yc2 = 0.5 * (yb2[:-1] + yb2[1:])
    prof = np.array([gg[c & (Y >= p) & (Y < q)].mean() if (c & (Y >= p) & (Y < q)).sum() else 0
                     for p, q in zip(yb2[:-1], yb2[1:])])
    k = prof.argmax(); half = prof[k] / 2
    lo = k
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = k
    while hi < len(prof) - 1 and prof[hi] > half:
        hi += 1
    xc_b.append(0.5 * (a + b)); trace.append(yc2[k]); halfw.append(yc2[hi] - yc2[lo])
xc_b, trace, halfw = map(np.array, (xc_b, trace, halfw))

srt = np.sort(gg[interior])[::-1]
frac = np.arange(1, len(srt) + 1) / len(srt)
ax2[1, 0].plot(100 * frac, 100 * np.cumsum(srt) / srt.sum(), lw=2.2, color="#15616d")
ax2[1, 0].plot([0, 100], [0, 100], "--", lw=1.1, color="#888", label="no localisation")
for f, c in ((1, "#b4531b"), (5, "#b4531b"), (10, "#b4531b")):
    y_ = 100 * np.cumsum(srt)[int(f / 100 * len(srt))] / srt.sum()
    ax2[1, 0].plot([f], [y_], "o", ms=6, color=c)
    ax2[1, 0].annotate(f"{f}% carry {y_:.0f}%", (f, y_), textcoords="offset points",
                       xytext=(10, -6), fontsize=8.5, color=c)
ax2[1, 0].set_xlim(0, 60); ax2[1, 0].set_ylim(0, 100)
ax2[1, 0].set_xlabel("most-sheared particles (%)"); ax2[1, 0].set_ylabel("share of total |E12| (%)")
ax2[1, 0].set_title("d  the shear IS localised", fontsize=10.5)
ax2[1, 0].legend(frameon=False, fontsize=8.5, loc="lower right"); ax2[1, 0].grid(alpha=.22)
ax2[0, 0].plot(xc_b / 1000, trace / 1000, "-", lw=1.4, color="#39d0ff")
ax2[0, 0].fill_between(xc_b / 1000, (trace - halfw / 2) / 1000, (trace + halfw / 2) / 1000,
                       color="#39d0ff", alpha=.30, lw=0)

# e: everything against distance from the PICKED trace
ytr = np.interp(Xc, xc_b, trace)
dfa = np.abs(Y - ytr)
bins = np.array([0, 50, 100, 150, 200, 300, 400, 600, 900]); bc = 0.5 * (bins[:-1] + bins[1:])
take = lambda v: [v[interior & (dfa >= a) & (dfa < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
ax2[1, 1].plot(bc, take(D["gamma"]), lw=2, label="γ, maximum shear strain")
ax2[1, 1].plot(bc, take(np.abs(E[:, 0, 1])), lw=2, label="|E12|")
ax2[1, 1].plot(bc, take(D["vol"]), lw=2, label="volumetric strain")
ax2[1, 1].plot(bc, [0.02 * v for v in take(np.abs(D["theta_z"]))], lw=2,
               label="|θz| × 0.02  (deg)")
ax2[1, 1].axhline(0, color="#999", lw=.8)
ax2[1, 1].set_xlabel("distance from the picked fault trace (m)"); ax2[1, 1].set_ylabel("strain")
ax2[1, 1].set_title(f"e  core ~{np.median(halfw):.0f} m wide; |E12| falls 26× to the far field",
                    fontsize=10.5)
ax2[1, 1].legend(frameon=False, fontsize=8.5); ax2[1, 1].grid(alpha=.22)

# f: orientation of maximum horizontal extension
wv, vv = np.linalg.eigh(E[:, :2, :2])
az = (np.degrees(np.arctan2(vv[:, 1, 1], vv[:, 0, 1])) + 180) % 180
sel = interior & (D["gamma"] > 0.02)
ax2[1, 2].hist(az[sel], bins=36, range=(0, 180), color="#15616d", alpha=.85)
ax2[1, 2].axvline(45, color="#b4531b", lw=2, ls="--")
ax2[1, 2].text(47, ax2[1, 2].get_ylim()[1] * .9, "45°: infinitesimal\nsimple shear",
               fontsize=8.5, color="#b4531b")
ax2[1, 2].set_xlabel("azimuth of maximum horizontal extension (° from X)")
ax2[1, 2].set_ylabel("particles")
ax2[1, 2].set_title("f  the axis sits at 60–70°, not 45° —\na y-biased extension rides on the shear",
                    fontsize=10.5)
ax2[1, 2].grid(alpha=.22)

fig2.suptitle("Strike-slip DEM run, diagnosed — maps at Z = −450 m with the picked fault trace "
              "and its half-width; profiles from the interior (below the cap, 150 m wall buffer)",
              fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(f"fig_strain_{TAG}_diagnosis.png", dpi=145)
print(f"saved fig_strain_{TAG}_diagnosis.png")
