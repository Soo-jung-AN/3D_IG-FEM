"""Figures for the three-way 3D comparison produced by compare_3d.py."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
EXT = [gx.min() / 1000, gx.max() / 1000, gz.min() / 1000, gz.max() / 1000]
msk = ~inside[:, jy, :].T          # (nz, nx) after transpose


def show(ax, field2d, **kw):
    """field2d comes in as (nx, nz); imshow wants (nz, nx)."""
    return ax.imshow(np.ma.array(field2d.T, mask=msk), origin="lower",
                     extent=EXT, aspect="equal", **kw)


# --- Figure 1: Vp and traveltime on the mid-model slice, all three routes
fig, ax = plt.subplots(3, 2, figsize=(16, 9))
vlo = min(np.percentile(v[:, jy, :][inside[:, jy, :]], 1) for v in vp)
vhi = max(np.percentile(v[:, jy, :][inside[:, jy, :]], 99) for v in vp)
tmax = max(np.ma.array(t[:, jy, :], mask=~inside[:, jy, :]).max() for t in tts)
for r, n in enumerate(names):
    im = show(ax[r, 0], vp[r][:, jy, :] / 1000, cmap="turbo", vmin=vlo / 1000, vmax=vhi / 1000)
    ax[r, 0].set_title(f"{n} — Vp (km/s)", fontsize=11)
    plt.colorbar(im, ax=ax[r, 0], shrink=0.85, pad=0.01)

    T = tts[r][:, jy, :]
    im2 = show(ax[r, 1], T, cmap="magma_r", vmin=0, vmax=tmax)
    ax[r, 1].contour(gx / 1000, gz / 1000, np.ma.array(T.T, mask=msk),
                     levels=np.arange(0, tmax, 2.0), colors="w", linewidths=0.6)
    ax[r, 1].plot(src[0] / 1000, src[2] / 1000, "*", ms=14, mfc="cyan", mec="k")
    ax[r, 1].set_title(f"{n} — first-arrival traveltime (s), 2 s contours", fontsize=11)
    plt.colorbar(im2, ax=ax[r, 1], shrink=0.85, pad=0.01)
    for c in (0, 1):
        ax[r, c].set_ylabel("Z (km)")
for c in (0, 1):
    ax[-1, c].set_xlabel("X (km)")
fig.suptitle(f"3D DEM → Vp → 3D eikonal traveltime, slice y = {gy[jy]/1000:.1f} km "
             "(the source ★ is a point source in the full 3D volume)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig("fig3d_vp_traveltime.png", dpi=140)
print("saved fig3d_vp_traveltime.png")

# --- Figure 2: surface traveltime curve + traveltime differences
fig2, ax2 = plt.subplots(1, 2, figsize=(15, 4.6))
surf = np.array([np.where(inside[ix, jy])[0].max() if inside[ix, jy].any() else -1
                 for ix in range(len(gx))])
ok = surf >= 0
for r, n in enumerate(names):
    ax2[0].plot(gx[ok] / 1000, tts[r][np.arange(len(gx))[ok], jy, surf[ok]], lw=1.6, label=n)
ax2[0].axvline(src[0] / 1000, color="k", ls=":", lw=1)
ax2[0].set_xlabel("X (km)"); ax2[0].set_ylabel("first arrival (s)")
ax2[0].set_title("Surface t–x curve along the slice", fontsize=11)
ax2[0].legend(frameon=False, fontsize=9); ax2[0].grid(alpha=0.25)

ref = names.index("IG-FEM + Botter")
dif = tts[names.index("SSPX + Botter")][:, jy, :] - tts[ref][:, jy, :]
lim = np.nanpercentile(np.abs(dif[inside[:, jy, :]]), 99)
im = show(ax2[1], dif, cmap="RdBu_r", vmin=-lim, vmax=lim)
ax2[1].set_title("Δt: (SSPX + Botter) − (IG-FEM + Botter), s", fontsize=11)
ax2[1].set_xlabel("X (km)"); ax2[1].set_ylabel("Z (km)")
plt.colorbar(im, ax=ax2[1], shrink=0.85, pad=0.01)
fig2.tight_layout()
fig2.savefig("fig3d_traveltime_curve.png", dpi=140)
print("saved fig3d_traveltime_curve.png")

# --- Figure 3: impedance and synthetic seismic sections
fig3, ax3 = plt.subplots(3, 2, figsize=(16, 9))
for r, n in enumerate(names):
    Z = vp[r][:, jy, :] * rho[r][:, jy, :]
    im = show(ax3[r, 0], Z / 1e6, cmap="cividis")
    ax3[r, 0].set_title(f"{n} — acoustic impedance (10⁶ kg m⁻² s⁻¹)", fontsize=11)
    plt.colorbar(im, ax=ax3[r, 0], shrink=0.85, pad=0.01)

    s = secs[r]
    lim = np.nanpercentile(np.abs(s[np.isfinite(s)]), 99)
    im2 = ax3[r, 1].imshow(np.ma.array(s.T, mask=msk | ~np.isfinite(s.T)),
                           origin="lower", extent=EXT, aspect="equal",
                           cmap="gray", vmin=-lim, vmax=lim)
    ax3[r, 1].set_title(f"{n} — synthetic section, 30 Hz", fontsize=11)
    plt.colorbar(im2, ax=ax3[r, 1], shrink=0.85, pad=0.01)
    for c in (0, 1):
        ax3[r, c].set_ylabel("Z (km)")
for c in (0, 1):
    ax3[-1, c].set_xlabel("X (km)")
fig3.suptitle("Botter et al. step 3 (lightweight form): impedance → 1D-convolution "
              "synthetic section, 30 Hz zero-phase Ricker", fontsize=12)
fig3.tight_layout(rect=[0, 0, 1, 0.955])
fig3.savefig("fig3d_seismic.png", dpi=140)
print("saved fig3d_seismic.png")

# --- Figure 4: the two strain estimates, and what they do to Vp
fig4, ax4 = plt.subplots(1, 3, figsize=(16, 4.4))
b = (np.abs(vol_s) < 1) & (np.abs(vol_i) < 1)
ax4[0].hist(vol_s[b], bins=200, range=(-1, 1), histtype="step", lw=1.5, label="SSPX")
ax4[0].hist(vol_i[b], bins=200, range=(-1, 1), histtype="step", lw=1.5, label="IG-FEM")
ax4[0].set_yscale("log"); ax4[0].set_xlabel("volumetric strain  det(F) − 1")
ax4[0].set_ylabel("particles"); ax4[0].legend(frameon=False, fontsize=9)
ax4[0].set_title("Volumetric strain distribution", fontsize=11)

h = ax4[1].hist2d(vol_s[b], vol_i[b], bins=200, range=[[-1, 1], [-1, 1]],
                  norm=matplotlib.colors.LogNorm(), cmap="viridis")
ax4[1].plot([-1, 1], [-1, 1], "w--", lw=1)
ax4[1].set_xlabel("SSPX"); ax4[1].set_ylabel("IG-FEM")
ax4[1].set_title(f"pointwise agreement, r = {np.corrcoef(vol_s[b], vol_i[b])[0,1]:+.3f}",
                 fontsize=11)
plt.colorbar(h[3], ax=ax4[1], shrink=0.85, pad=0.01)

for r, n in enumerate(names):
    v = vp[r][:, jy, :]
    prof = np.array([np.ma.array(v[:, k], mask=~inside[:, jy, k]).mean()
                     for k in range(len(gz))])
    ax4[2].plot(prof / 1000, gz / 1000, lw=1.8, label=n)
ax4[2].set_xlabel("slice-mean Vp (km/s)"); ax4[2].set_ylabel("Z (km)")
ax4[2].set_title("Depth trend of Vp", fontsize=11)
ax4[2].legend(frameon=False, fontsize=9); ax4[2].grid(alpha=0.25)
fig4.tight_layout()
fig4.savefig("fig3d_strain_vp.png", dpi=140)
print("saved fig3d_strain_vp.png")
