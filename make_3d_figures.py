"""Figures for the three-way 3D comparison produced by compare_3d.py.

The model is 170 km wide and 14 km deep, so every cross-section is drawn
with an explicit 3x vertical exaggeration -- stated in each title, since a
silently stretched section is the easiest way to make a velocity model
look more structured than it is.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

VE = 3.0            # vertical exaggeration of every cross-section
DEPTH_SLICE = -9.0  # km, depth of the map-view slice

D = np.load("./results/compare_3d.npz")
gx, gy, gz = D["gx"], D["gy"], D["gz"]
inside, jy, src = D["inside"], int(D["slice_y"]), D["src"]
names = [str(n) for n in D["names"]]
vp, rho, tts, secs = D["vp_grids"], D["rho_grids"], D["tts"], D["secs"]
vol_s, vol_i = D["vol_sspx"], D["vol_igfem"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})
XZ = [gx.min() / 1000, gx.max() / 1000, gz.min() / 1000, gz.max() / 1000]
XY = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]
in_xz = inside[:, jy, :]
msk = ~in_xz.T                       # (nz, nx), ready for imshow


def xz(ax, field, **kw):
    """field arrives as (nx, nz); imshow wants (nz, nx)."""
    ax.set_xlabel("X (km)"); ax.set_ylabel("Z (km)")
    return ax.imshow(np.ma.array(field.T, mask=msk), origin="lower",
                     extent=XZ, aspect=VE, **kw)


def stack(nrow=3, h=3.4):
    fig, ax = plt.subplots(nrow, 1, figsize=(15, h * nrow))
    return fig, np.atleast_1d(ax)


# --- Figure 1: Vp on the mid-model slice -------------------------------
fig, ax = stack()
lo = min(np.percentile(vp[r][:, jy, :][in_xz], 1) for r in (0, 1)) / 1000
hi = max(np.percentile(vp[r][:, jy, :][in_xz], 99) for r in (0, 1)) / 1000
for r, n in enumerate(names):
    v = vp[r][:, jy, :] / 1000
    kw = dict(vmin=lo, vmax=hi) if r < 2 else dict(
        vmin=np.percentile(v[in_xz], 1), vmax=np.percentile(v[in_xz], 99))
    im = xz(ax[r], v, cmap="turbo", **kw)
    ax[r].set_title(f"{n} — Vp (km/s){'' if r < 2 else '   [own colour scale]'}", fontsize=11)
    plt.colorbar(im, ax=ax[r], shrink=0.9, pad=0.008)
fig.suptitle(f"Vp from three rock-physics routes on the same 3D DEM deformation — "
             f"slice y = {gy[jy]/1000:.1f} km, {VE:.0f}× vertical exaggeration", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig3d_vp.png", dpi=140)
print("saved fig3d_vp.png")

# --- Figure 2: 3D eikonal traveltime on the same slice -----------------
fig2, ax2 = stack()
tmax = max(np.ma.array(t[:, jy, :], mask=~in_xz).max() for t in tts)
for r, n in enumerate(names):
    T = tts[r][:, jy, :]
    im = xz(ax2[r], T, cmap="magma_r", vmin=0, vmax=tmax)
    ax2[r].contour(gx / 1000, gz / 1000, np.ma.array(T.T, mask=msk),
                   levels=np.arange(2, tmax, 2.0), colors="w", linewidths=0.7)
    ax2[r].plot(src[0] / 1000, src[2] / 1000, "*", ms=16, mfc="cyan", mec="k")
    ax2[r].set_title(f"{n} — first-arrival traveltime (s), 2 s contours", fontsize=11)
    plt.colorbar(im, ax=ax2[r], shrink=0.9, pad=0.008)
fig2.suptitle("3D eikonal first arrivals from one point source (★) in the full volume — "
              f"slice y = {gy[jy]/1000:.1f} km, {VE:.0f}× vertical exaggeration", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig("fig3d_traveltime.png", dpi=140)
print("saved fig3d_traveltime.png")

# --- Figure 3: surface t-x curve, and where the two strain routes differ
fig3, ax3 = plt.subplots(2, 1, figsize=(15, 8))
surf = np.array([np.where(inside[ix, jy])[0].max() if inside[ix, jy].any() else -1
                 for ix in range(len(gx))])
ok = surf >= 0
ix_ok = np.arange(len(gx))[ok]
for r, n in enumerate(names):
    ax3[0].plot(gx[ok] / 1000, tts[r][ix_ok, jy, surf[ok]], lw=1.6, label=n)
ax3[0].axvline(src[0] / 1000, color="k", ls=":", lw=1)
ax3[0].set_xlabel("X (km)"); ax3[0].set_ylabel("first arrival (s)")
ax3[0].set_title("Surface t–x curve along the slice (receivers on the free surface)", fontsize=11)
ax3[0].legend(frameon=False, fontsize=9); ax3[0].grid(alpha=0.25)

ref = names.index("IG-FEM + Botter")
dif = tts[names.index("SSPX + Botter")][:, jy, :] - tts[ref][:, jy, :]
lim = np.nanpercentile(np.abs(dif[in_xz]), 99)
im = xz(ax3[1], dif, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax3[1].set_title(f"Δt = (SSPX + Botter) − (IG-FEM + Botter), s   "
                 f"[RMS {np.sqrt(np.mean((dif[in_xz])**2)):.3f} s over the whole 3D volume slice]",
                 fontsize=11)
plt.colorbar(im, ax=ax3[1], shrink=0.9, pad=0.008)
fig3.tight_layout()
fig3.savefig("fig3d_traveltime_curve.png", dpi=140)
print("saved fig3d_traveltime_curve.png")

# --- Figure 4: impedance and 1D-convolution synthetic sections ---------
fig4, ax4 = plt.subplots(3, 2, figsize=(17, 9))
for r, n in enumerate(names):
    Z = vp[r][:, jy, :] * rho[r][:, jy, :] / 1e6
    ax4[r, 0].set_xlabel("X (km)"); ax4[r, 0].set_ylabel("Z (km)")
    im = ax4[r, 0].imshow(np.ma.array(Z.T, mask=msk), origin="lower", extent=XZ,
                          aspect=VE, cmap="cividis")
    ax4[r, 0].set_title(f"{n} — impedance (10⁶ kg m⁻² s⁻¹)", fontsize=10)
    plt.colorbar(im, ax=ax4[r, 0], shrink=0.85, pad=0.01)

    s = secs[r]
    lim = np.nanpercentile(np.abs(s[np.isfinite(s)]), 99)
    ax4[r, 1].set_xlabel("X (km)"); ax4[r, 1].set_ylabel("Z (km)")
    im2 = ax4[r, 1].imshow(np.ma.array(s.T, mask=msk | ~np.isfinite(s.T)),
                           origin="lower", extent=XZ, aspect=VE,
                           cmap="gray", vmin=-lim, vmax=lim)
    ax4[r, 1].set_title(f"{n} — synthetic section, 30 Hz", fontsize=10)
    plt.colorbar(im2, ax=ax4[r, 1], shrink=0.85, pad=0.01)
fig4.suptitle("Botter et al. step 3 (lightweight form): impedance → 1D-convolution synthetic "
              f"section, 30 Hz zero-phase Ricker, {VE:.0f}× vertical exaggeration", fontsize=12)
fig4.tight_layout(rect=[0, 0, 1, 0.955])
fig4.savefig("fig3d_seismic.png", dpi=140)
print("saved fig3d_seismic.png")

# --- Figure 5: map view, the thing a 2D model cannot produce -----------
kz = np.argmin(np.abs(gz / 1000 - DEPTH_SLICE))
fig5, ax5 = plt.subplots(3, 1, figsize=(14, 10))
in_xy = inside[:, :, kz]
lo = min(np.percentile(vp[r][:, :, kz][in_xy], 1) for r in (0, 1)) / 1000
hi = max(np.percentile(vp[r][:, :, kz][in_xy], 99) for r in (0, 1)) / 1000
for r, n in enumerate(names):
    v = vp[r][:, :, kz] / 1000
    kw = dict(vmin=lo, vmax=hi) if r < 2 else dict(
        vmin=np.percentile(v[in_xy], 1), vmax=np.percentile(v[in_xy], 99))
    im = ax5[r].imshow(np.ma.array(v.T, mask=~in_xy.T), origin="lower", extent=XY,
                       aspect="equal", cmap="turbo", **kw)
    ax5[r].set_xlabel("X (km)"); ax5[r].set_ylabel("Y (km)")
    ax5[r].set_title(f"{n} — Vp (km/s) at Z = {gz[kz]/1000:.1f} km"
                     f"{'' if r < 2 else '   [own colour scale]'}", fontsize=11)
    plt.colorbar(im, ax=ax5[r], shrink=0.9, pad=0.008)
fig5.suptitle("Map view — the along-strike variation a 2D model cannot produce", fontsize=12)
fig5.tight_layout(rect=[0, 0, 1, 0.965])
fig5.savefig("fig3d_mapview.png", dpi=140)
print("saved fig3d_mapview.png")

# --- Figure 6: the two strain estimates, and what they do to Vp --------
fig6, ax6 = plt.subplots(1, 3, figsize=(16, 4.4))
b = (np.abs(vol_s) < 1) & (np.abs(vol_i) < 1)
ax6[0].hist(vol_s[b], bins=200, range=(-1, 1), histtype="step", lw=1.5, label="SSPX")
ax6[0].hist(vol_i[b], bins=200, range=(-1, 1), histtype="step", lw=1.5, label="IG-FEM")
ax6[0].set_yscale("log"); ax6[0].set_xlabel("volumetric strain  det(F) − 1")
ax6[0].set_ylabel("particles"); ax6[0].legend(frameon=False, fontsize=9)
ax6[0].set_title("Volumetric strain distribution", fontsize=11)

h = ax6[1].hist2d(vol_s[b], vol_i[b], bins=200, range=[[-1, 1], [-1, 1]],
                  norm=LogNorm(), cmap="viridis")
ax6[1].plot([-1, 1], [-1, 1], "w--", lw=1)
ax6[1].set_xlabel("SSPX"); ax6[1].set_ylabel("IG-FEM")
ax6[1].set_title(f"pointwise agreement, r = {np.corrcoef(vol_s[b], vol_i[b])[0,1]:+.3f} "
                 f"(n = {b.sum():,})", fontsize=11)
plt.colorbar(h[3], ax=ax6[1], shrink=0.85, pad=0.01)

for r, n in enumerate(names):
    prof = np.array([np.ma.array(vp[r][:, jy, k], mask=~inside[:, jy, k]).mean()
                     for k in range(len(gz))])
    ax6[2].plot(prof / 1000, gz / 1000, lw=1.8, label=n)
ax6[2].set_xlabel("slice-mean Vp (km/s)"); ax6[2].set_ylabel("Z (km)")
ax6[2].set_title("Depth trend of Vp", fontsize=11)
ax6[2].legend(frameon=False, fontsize=9); ax6[2].grid(alpha=0.25)
fig6.tight_layout()
fig6.savefig("fig3d_strain_vp.png", dpi=140)
print("saved fig3d_strain_vp.png")
