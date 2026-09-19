"""Figures for plausibility_3d.py: the two calibrations side by side."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rock_physics as R

VE = 3.0
CP, CC, CR = "#b4531b", "#1f6fb4", "#5a6472"      # published, crustal, reference

D = np.load("./results/plausibility_3d.npz")
gx, gy, gz = D["gx"], D["gy"], D["gz"]
inside, jy, dx, src = D["inside"], int(D["slice_y"]), float(D["dx"]), D["src"]
in_xz = inside[:, jy, :]
msk = ~in_xz.T
XZ = [gx.min() / 1000, gx.max() / 1000, gz.min() / 1000, gz.max() / 1000]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})

X0 = np.loadtxt("./txt/init_pos.txt")
vol = np.load("./results/compare_3d.npz")["vol_igfem"]
Z = X0[:, 2]

cases = {}
for key, props, vsf in (("published", R.zoned_initial_properties, R.vs_from_vp),
                        ("crustal", R.crustal_initial_properties, R.vs_from_vp_brocher)):
    p0, rg, v0 = props(X0)
    phi = R.porosity_from_strain(vol, p0)
    rho = R.density_from_porosity(phi, rg)
    Vp = R.vp_from_strain(vol, v0)
    cases[key] = dict(phi=phi, rho=rho, Vp=Vp, Vs=vsf(Vp))

# --- Figure 1: depth profiles + Poisson --------------------------------
edges = np.arange(-15e3, 1, 500.0)
mid = 0.5 * (edges[:-1] + edges[1:]) / 1000
prof = lambda v: np.array([v[(Z >= a) & (Z < b)].mean() for a, b in zip(edges[:-1], edges[1:])])

fig, ax = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
ref = {  # Brocher (2005); Christensen & Mooney (1995), for a 15 km column
    0: [(-7, 0, 2.5, 4.5), (-11, -7, 4.5, 5.5), (-15, -11, 5.8, 6.3)],
    1: [(-7, 0, 2200, 2500), (-11, -7, 2500, 2650), (-15, -11, 2700, 2800)],
    2: [(-7, 0, 5.5, 11.3), (-11, -7, 11.3, 14.6), (-15, -11, 15.7, 17.6)],
    3: [(-15, 0, 1.65, 2.00)],
}
for k, (title, unit) in enumerate([("Vp", "km/s"), ("bulk density", "kg/m³"),
                                   ("impedance", "10⁶ kg m⁻² s⁻¹"), ("Vp/Vs", "")]):
    for zlo, zhi, vlo, vhi in ref[k]:
        ax[k].fill_betweenx([zlo, zhi], vlo, vhi, color=CR, alpha=.16, lw=0)
    for key, c in (("published", CP), ("crustal", CC)):
        d = cases[key]
        v = {0: d["Vp"], 1: d["rho"], 2: d["rho"] * d["Vp"] / 1e3,
             3: d["Vp"] / d["Vs"]}[k]
        ax[k].plot(prof(v), mid, lw=2, color=c, label=key)
    ax[k].set_xlabel(f"{title}" + (f"  ({unit})" if unit else ""))
    ax[k].grid(alpha=.25)
ax[0].set_ylabel("Z (km)")
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
ax[3].set_xlim(1.4, 4.0)
fig.suptitle("Depth profiles against a real 15 km crustal column (shaded: Brocher 2005; "
             "Christensen & Mooney 1995)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig3d_plaus_profiles.png", dpi=140)
print("saved fig3d_plaus_profiles.png")

# --- Figure 2: Poisson's ratio distribution ---------------------------
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.2))
for key, c in (("published", CP), ("crustal", CC)):
    d = cases[key]
    r = d["Vp"] / d["Vs"]
    nu = (r ** 2 - 2) / (2 * (r ** 2 - 1))
    ax2[0].hist(nu, bins=180, range=(0, .5), histtype="step", lw=1.8, color=c,
                label=f"{key}  (median {np.median(nu):.3f})")
ax2[0].axvspan(0.40, 0.50, color="#96341f", alpha=.10, lw=0)
ax2[0].text(0.445, ax2[0].get_ylim()[1] * .82, "near-fluid", ha="center", fontsize=9, color="#96341f")
ax2[0].set_xlabel("Poisson's ratio ν"); ax2[0].set_ylabel("particles")
ax2[0].legend(frameon=False, fontsize=9); ax2[0].set_title("Poisson's ratio", fontsize=11)
ax2[0].grid(alpha=.25)

v = np.linspace(1.2, 8.0, 600)
for fn, c, lab in ((R.vs_from_vp, CP, "Han (1986), Botter Eq. (4)"),
                   (R.vs_from_vp_brocher, CC, "Brocher (2005) regression fit")):
    with np.errstate(divide="ignore"):
        ax2[1].plot(v, v / fn(v), lw=2, color=c, label=lab)
ax2[1].axvspan(3.0, 5.5, color=CR, alpha=.15, lw=0)
ax2[1].text(4.25, 4.6, "Han's calibration range", ha="center", fontsize=9, color=CR)
ax2[1].axhline(2.45, color="#96341f", lw=.9, ls="--")
ax2[1].text(7.9, 2.55, "ν = 0.40", ha="right", fontsize=9, color="#96341f")
ax2[1].set_ylim(1.4, 5.2); ax2[1].set_xlim(1.2, 8.0)
# where each calibration's velocities actually sit
tw = ax2[1].twinx()
for key, c in (("published", CP), ("crustal", CC)):
    tw.hist(cases[key]["Vp"], bins=140, range=(1.2, 8.0), histtype="stepfilled",
            color=c, alpha=.18, lw=0)
tw.set_yticks([]); tw.set_ylim(0, tw.get_ylim()[1] * 3.2)
ax2[1].set_xlabel("Vp (km/s)"); ax2[1].set_ylabel("Vp/Vs")
ax2[1].set_title("Both relations blow up at low Vp — the published\ncalibration (shaded) lives there",
                 fontsize=11)
ax2[1].legend(frameon=False, fontsize=9, loc="upper right"); ax2[1].grid(alpha=.25)
ax2[1].set_zorder(tw.get_zorder() + 1); ax2[1].patch.set_visible(False)
fig2.tight_layout()
fig2.savefig("fig3d_plaus_vpvs.png", dpi=140)
print("saved fig3d_plaus_vpvs.png")

# --- Figure 3: sections, 30 Hz vs 5 Hz, both calibrations -------------
fig3, ax3 = plt.subplots(2, 2, figsize=(17, 6.4))
for r, key in enumerate(["published", "crustal"]):
    for c, f in enumerate([30, 5]):
        s = D[f"sec_{key}_{f}"]
        lim = np.nanpercentile(np.abs(s[np.isfinite(s)]), 99)
        ax3[r, c].imshow(np.ma.array(s.T, mask=msk | ~np.isfinite(s.T)), origin="lower",
                         extent=XZ, aspect=VE, cmap="gray", vmin=-lim, vmax=lim)
        ok = "grid-supported" if f == 5 else "below the grid — this is texture, not structure"
        ax3[r, c].set_title(f"{key} calibration — {f} Hz   [{ok}]", fontsize=10)
        ax3[r, c].set_xlabel("X (km)"); ax3[r, c].set_ylabel("Z (km)")
fig3.suptitle("A 250 m grid resolves λ/4 = 250 m. At 30 Hz the two calibrations correlate at "
              "+0.993 despite a 60% velocity difference; at 5 Hz, +0.609.", fontsize=11)
fig3.tight_layout(rect=[0, 0, 1, 0.94])
fig3.savefig("fig3d_plaus_sections.png", dpi=140)
print("saved fig3d_plaus_sections.png")

# --- Figure 4: surface t-x -------------------------------------------
fig4, ax4 = plt.subplots(figsize=(13, 4.2))
surf = np.array([np.where(inside[i, jy])[0].max() if inside[i, jy].any() else -1
                 for i in range(len(gx))])
ok = surf >= 0
ix = np.arange(len(gx))[ok]
for key, c in (("published", CP), ("crustal", CC)):
    ax4.plot(gx[ok] / 1000, D[f"tt_{key}"][ix, jy, surf[ok]], lw=1.8, color=c, label=key)
off = np.abs(gx[ok] - src[0]) / 1000
for v, ls in ((6.0, "--"), (4.0, ":")):
    ax4.plot(gx[ok] / 1000, off / v, ls, lw=1.2, color=CR,
             label=f"straight ray at {v:.0f} km/s")
ax4.axvline(src[0] / 1000, color="k", ls=":", lw=1)
ax4.set_xlabel("X (km)"); ax4.set_ylabel("first arrival (s)")
ax4.set_title("Surface t–x curve: the published calibration is ~60% too slow for a crustal column",
              fontsize=11)
ax4.legend(frameon=False, fontsize=9); ax4.grid(alpha=.25)
fig4.tight_layout()
fig4.savefig("fig3d_plaus_traveltime.png", dpi=140)
print("saved fig3d_plaus_traveltime.png")
