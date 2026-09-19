"""Figures for vp_from_strain.py, laid out for the model's own geometry.

For a strike-slip run the informative view is the map view: the fault
zone is a band in y, and slip is in x. A single x-z section at mid-y
lies IN the fault plane and shows almost nothing of the structure.
"""
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
DEPTHS = (-150.0, -450.0, -800.0)

D = np.load(NPZ)
X0, X1, vol = D["X0"], D["X1"], D["vol"]
Vp, Vp_ini, rho_0 = D["Vp"], D["Vp_ini"], D["rho_0"]
dVp = 100.0 * (Vp / Vp_ini - 1.0)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})
tree = cKDTree(X1)
ax_grid = lambda k: np.arange(X1[:, k].min(), X1[:, k].max() + DX, DX)
gx, gy, gz = (ax_grid(k) for k in range(3))


def slab(fixed_axis, value, a_vals, b_vals):
    """Interpolate onto a plane through `value` on `fixed_axis`."""
    A, B = np.meshgrid(a_vals, b_vals, indexing="ij")
    cols = [None, None, None]
    free = [k for k in range(3) if k != fixed_axis]
    cols[free[0]], cols[free[1]] = A.ravel(), B.ravel()
    cols[fixed_axis] = np.full(A.size, value)
    nodes = np.column_stack(cols)
    dist, idx = tree.query(nodes, k=6)
    w = 1.0 / np.maximum(dist, 1e-9) ** 2
    mask = (dist[:, 0] >= 2 * DX).reshape(A.shape)
    return (lambda v: np.ma.array(((w * v[idx]).sum(1) / w.sum(1)).reshape(A.shape).T,
                                  mask=mask.T)), A.shape


def show(ax, field, ext, title, xlab, ylab, **kw):
    im = ax.imshow(field, origin="lower", extent=ext, aspect="equal", **kw)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    plt.colorbar(im, ax=ax, shrink=0.86, pad=0.012)


fig, ax = plt.subplots(2, 3, figsize=(16, 9))
EXT_XY = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]

for j, zc in enumerate(DEPTHS):
    g, _ = slab(2, zc, gx, gy)
    show(ax[0, j], g(dVp), EXT_XY,
         f"{'abc'[j]}  ΔVp from strain at Z = {zc:.0f} m  (%)",
         "X (km)", "Y (km)", cmap="RdBu_r", vmin=-25, vmax=25)

# fault-perpendicular section, at mid-x
x_mid = 0.5 * (gx.min() + gx.max())
g, _ = slab(0, x_mid, gy, gz)
EXT_YZ = [gy.min() / 1000, gy.max() / 1000, gz.min() / 1000, gz.max() / 1000]
show(ax[1, 0], g(Vp), EXT_YZ, f"d  Vp across strike, X = {x_mid/1000:.2f} km  (km/s)",
     "Y (km)", "Z (km)", cmap="turbo", vmin=np.percentile(Vp, 1), vmax=np.percentile(Vp, 99))
show(ax[1, 1], g(dVp), EXT_YZ, "e  ΔVp from strain, same section  (%)",
     "Y (km)", "Z (km)", cmap="RdBu_r", vmin=-25, vmax=25)

a = ax[1, 2]
ev = np.linspace(-1, 1, 400)
a.plot(ev, 100 * (R.vp_from_strain(ev, 1.0) - 1), lw=2.2, color="#15616d",
       label="Botter et al. (2014) Eq. (3)")
a.axhline(0, color="#999", lw=.8); a.axvline(0, color="#999", lw=.8)
a.set_xlim(-0.5, 0.8); a.set_ylim(-28, 18)
a.set_xlabel("volumetric strain  det(F) − 1"); a.set_ylabel("ΔVp  (%)")
a.set_title("f  the transfer function, over this model's strain", fontsize=10.5)
tw = a.twinx()
tw.hist(vol[np.abs(vol) < 1], bins=220, range=(-0.5, 0.8), color="#b4531b", alpha=.22, lw=0)
tw.set_yticks([]); tw.set_ylim(0, tw.get_ylim()[1] * 3.0)
a.set_zorder(tw.get_zorder() + 1); a.patch.set_visible(False)
a.legend(frameon=False, fontsize=9, loc="lower left"); a.grid(alpha=.22)
a.text(.50, -23, "dilatation slows\nthe rock", fontsize=8.5, color="#15616d", ha="center")
a.text(-.34, 11, "compaction\nspeeds it up", fontsize=8.5, color="#15616d", ha="center")

fig.suptitle("Vp estimated from the volumetric strain of a strike-slip DEM run — "
             f"{len(X0):,} particles, 1.97 × 1.97 × 0.97 km", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(f"fig_vp_{TAG}.png", dpi=145)
print(f"saved fig_vp_{TAG}.png")

# --- detail: reference state, estimator agreement, Vp distribution -----
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.2))
for v in np.unique(rho_0):
    m = rho_0 == v
    ax2[0].scatter(Vp[m][::40], X0[m, 2][::40] / 1000, s=3, alpha=.25, label=f"ρ = {v:.0f} kg/m³")
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

# dilatated fraction across the fault, by depth band
yb = np.linspace(X0[:, 1].min(), X0[:, 1].max(), 26)
yc = 0.5 * (yb[:-1] + yb[1:]) / 1000
for zlo, zhi, lab in ((-300, 0, "0 to −300 m"), (-600, -300, "−300 to −600 m"),
                      (-1000, -600, "−600 to −1000 m")):
    m = (X0[:, 2] >= zlo) & (X0[:, 2] < zhi)
    frac = [100 * np.mean(vol[m & (X0[:, 1] >= p) & (X0[:, 1] < q)] > 0.05)
            for p, q in zip(yb[:-1], yb[1:])]
    ax2[2].plot(yc, frac, lw=1.9, label=lab)
ax2[2].set_xlabel("Y (km)"); ax2[2].set_ylabel("particles dilated by > 5%  (%)")
ax2[2].set_title("the fault zone, seen as dilatation across strike", fontsize=10.5)
ax2[2].legend(frameon=False, fontsize=8.5); ax2[2].grid(alpha=.22)

fig2.tight_layout()
fig2.savefig(f"fig_vp_{TAG}_detail.png", dpi=145)
print(f"saved fig_vp_{TAG}_detail.png")
