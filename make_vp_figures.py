"""Figures for vp_from_strain.py: strain in, Vp out."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import rock_physics as R

NPZ = sys.argv[1] if len(sys.argv) > 1 else "./results/vp_model3.npz"
TAG = sys.argv[2] if len(sys.argv) > 2 else "model3"
DX = 15.0

D = np.load(NPZ)
X0, X1, vol = D["X0"], D["X1"], D["vol"]
Vp, Vp_ini, rho, rho_0 = D["Vp"], D["Vp_ini"], D["rho"], D["rho_0"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})

gx = np.arange(X1[:, 0].min(), X1[:, 0].max() + DX, DX)
gz = np.arange(X1[:, 2].min(), X1[:, 2].max() + DX, DX)
y_mid = 0.5 * (X1[:, 1].min() + X1[:, 1].max())
GX, GZ = np.meshgrid(gx, gz, indexing="ij")
nodes = np.column_stack([GX.ravel(), np.full(GX.size, y_mid), GZ.ravel()])
tree = cKDTree(X1)
dist, idx = tree.query(nodes, k=6)
w = 1.0 / np.maximum(dist, 1e-9) ** 2
inside = (dist[:, 0] < 2 * DX).reshape(GX.shape)
grid = lambda v: ((w * v[idx]).sum(1) / w.sum(1)).reshape(GX.shape)
EXT = [gx.min() / 1000, gx.max() / 1000, gz.min() / 1000, gz.max() / 1000]
msk = ~inside.T


def panel(ax, field, title, **kw):
    im = ax.imshow(np.ma.array(grid(field).T, mask=msk), origin="lower", extent=EXT,
                   aspect="equal", **kw)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Z (km)")
    plt.colorbar(im, ax=ax, shrink=0.88, pad=0.012)
    return im


fig, ax = plt.subplots(2, 2, figsize=(14, 7.4))

lim = np.percentile(np.abs(vol[np.abs(vol) < 1]), 98)
panel(ax[0, 0], np.clip(vol, -1, 1), "a  volumetric strain  det(F) − 1",
      cmap="RdBu_r", vmin=-lim, vmax=lim)
panel(ax[0, 1], Vp, "b  Vp (km/s)", cmap="turbo",
      vmin=np.percentile(Vp, 1), vmax=np.percentile(Vp, 99))
panel(ax[1, 0], 100 * (Vp / Vp_ini - 1), "c  what the strain contributes:  Vp / Vp_ini − 1  (%)",
      cmap="RdBu_r", vmin=-25, vmax=25)

# the transfer function itself, with the data's own strain distribution under it
a = ax[1, 1]
ev = np.linspace(-1, 1, 400)
a.plot(ev, 100 * (R.vp_from_strain(ev, 1.0) - 1), lw=2.2, color="#15616d",
       label="Botter et al. (2014) Eq. (3)")
a.axhline(0, color="#999", lw=.8); a.axvline(0, color="#999", lw=.8)
a.set_xlim(-0.6, 0.8); a.set_ylim(-28, 20)
a.set_xlabel("volumetric strain  det(F) − 1"); a.set_ylabel("ΔVp  (%)")
a.set_title("d  the transfer function, over this model's strain", fontsize=10.5)
tw = a.twinx()
tw.hist(vol[np.abs(vol) < 1], bins=220, range=(-0.6, 0.8), color="#b4531b", alpha=.22, lw=0)
tw.set_yticks([]); tw.set_ylim(0, tw.get_ylim()[1] * 3.0)
a.set_zorder(tw.get_zorder() + 1); a.patch.set_visible(False)
a.legend(frameon=False, fontsize=9, loc="lower left"); a.grid(alpha=.22)
a.text(.52, -24, "dilatation\nslows the rock", fontsize=8.5, color="#15616d", ha="center")
a.text(-.42, 13, "compaction\nspeeds it up", fontsize=8.5, color="#15616d", ha="center")

fig.suptitle(f"Vp estimated from the volumetric strain of a discrete-element run — "
             f"{len(X0):,} particles, slice at y = {y_mid/1000:.2f} km", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(f"fig_vp_{TAG}.png", dpi=145)
print(f"saved fig_vp_{TAG}.png")

# --- second figure: the reference state and the two strain estimators ----
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.2))

for v in np.unique(rho_0):
    m = rho_0 == v
    ax2[0].scatter(Vp[m][::40], X0[m, 2][::40] / 1000, s=3, alpha=.25,
                   label=f"ρ = {v:.0f} kg/m³")
edges = np.arange(X0[:, 2].min(), X0[:, 2].max() + 20, 20.0)
mid = 0.5 * (edges[:-1] + edges[1:]) / 1000
prof = np.array([Vp[(X0[:, 2] >= a_) & (X0[:, 2] < b_)].mean() for a_, b_ in zip(edges[:-1], edges[1:])])
ax2[0].plot(prof, mid, lw=2.2, color="#16191c", label="layer mean")
ax2[0].set_xlabel("Vp (km/s)"); ax2[0].set_ylabel("initial Z (km)")
ax2[0].set_title("Vp against depth, coloured by the model's density", fontsize=10.5)
ax2[0].legend(frameon=False, fontsize=8, markerscale=3); ax2[0].grid(alpha=.22)

b = (np.abs(vol) < 1) & (np.abs(D["vol_sspx"]) < 1)
h = ax2[1].hist2d(D["vol_sspx"][b], vol[b], bins=180, range=[[-.4, .6], [-.4, .6]],
                  norm=matplotlib.colors.LogNorm(), cmap="viridis")
ax2[1].plot([-.4, .6], [-.4, .6], "w--", lw=1)
ax2[1].set_xlabel("SSPX  det(F) − 1"); ax2[1].set_ylabel("IG-FEM  det(F) − 1")
ax2[1].set_title(f"two strain estimators, r = {np.corrcoef(D['vol_sspx'][b], vol[b])[0,1]:+.3f}",
                 fontsize=10.5)
plt.colorbar(h[3], ax=ax2[1], shrink=.88, pad=.012)

ax2[2].hist(Vp, bins=160, color="#15616d", alpha=.8)
for v in np.unique(D["Vp_ini"]):
    ax2[2].axvline(v, color="#b4531b", ls="--", lw=1.1)
ax2[2].set_xlabel("Vp (km/s)"); ax2[2].set_ylabel("particles")
ax2[2].set_title("Vp distribution; dashed = the four reference velocities", fontsize=10.5)
ax2[2].grid(alpha=.22)

fig2.tight_layout()
fig2.savefig(f"fig_vp_{TAG}_detail.png", dpi=145)
print(f"saved fig_vp_{TAG}_detail.png")
