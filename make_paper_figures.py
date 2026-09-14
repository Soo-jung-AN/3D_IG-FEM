"""Figures for paper_calibration.py: the DEM's own constants, judged."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rock_physics as R

VE = 3.0
CP, CN, CD, CR = "#b4531b", "#1f6fb4", "#2e8b57", "#5a6472"   # published, Nafe-Drake, DEM-E, reference

D = np.load("./results/paper_calibration.npz")
gx, gy, gz = D["gx"], D["gy"], D["gz"]
inside, jy, dx = D["inside"], int(D["slice_y"]), float(D["dx"])
msk = ~inside[:, jy, :].T
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
p0, rg, v0 = R.zoned_initial_properties(X0)
_, rho, Vp, Vs, _ = R.synthesize_vpvs(vol, p0, rg, v0)
cases["as published"] = (Vp, Vs, rho, CP)
p0, rg, v0 = R.crustal_initial_properties(X0)
_, rho, Vp, Vs, _ = R.synthesize_vpvs(vol, p0, rg, v0)
cases["Nafe–Drake from ρ"] = (Vp, R.vs_from_vp_brocher(Vp), rho, CN)
rho, Vp, Vs = R.synthesize_paper(vol, X0)
cases["Table 1 E and ρ"] = (Vp, Vs, rho, CD)

# --- Figure 1: depth profiles, three calibrations ---------------------
edges = np.arange(-15e3, 1, 500.0)
mid = 0.5 * (edges[:-1] + edges[1:]) / 1000
prof = lambda v: np.array([v[(Z >= a) & (Z < b)].mean() for a, b in zip(edges[:-1], edges[1:])])

ref = {0: [(-7, 0, 3.0, 5.0), (-11, -7, 4.5, 5.8), (-15, -11, 5.8, 6.4)],
       1: [(-7, 0, 2200, 2500), (-11, -7, 2500, 2700), (-15, -11, 2650, 2800)],
       2: [(-7, 0, 6.6, 12.5), (-11, -7, 11.3, 15.7), (-15, -11, 15.4, 17.9)],
       3: [(-15, 0, 1.65, 2.00)]}
fig, ax = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
for k, (title, unit) in enumerate([("Vp", "km/s"), ("bulk density", "kg/m³"),
                                   ("impedance", "10⁶ kg m⁻² s⁻¹"), ("Vp/Vs", "")]):
    for zlo, zhi, vlo, vhi in ref[k]:
        ax[k].fill_betweenx([zlo, zhi], vlo, vhi, color=CR, alpha=.16, lw=0)
    for name, (Vp, Vs, rho, c) in cases.items():
        v = {0: Vp, 1: rho, 2: rho * Vp / 1e3, 3: Vp / Vs}[k]
        ax[k].plot(prof(v), mid, lw=2, color=c, label=name)
    ax[k].set_xlabel(title + (f"  ({unit})" if unit else ""))
    ax[k].grid(alpha=.25)
ax[0].set_ylabel("Z (km)")
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
ax[3].set_xlim(1.4, 3.2)
ax[3].axvline(np.sqrt(3), color=CD, ls=":", lw=1)
ax[3].annotate("√3, i.e. ν = 0.25\n(assumed — Table 1\ndoes not give ν)", (np.sqrt(3), -12.4),
               xytext=(2.30, -11.0), fontsize=8, color=CD,
               arrowprops=dict(arrowstyle="->", color=CD, lw=1))
fig.suptitle("Three calibrations of the same strain field. Shaded: a real 15 km crustal column "
             "(Brocher 2005; Christensen & Mooney 1995). The dip at −13 to −15 km is the décollement.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig3d_paper_profiles.png", dpi=140)
print("saved fig3d_paper_profiles.png")

# --- Figure 2: the layer stack audited -------------------------------
rows = R.paper_layer_summary()
names = [r["name"] for r in rows]
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) E vs UCS, with the modulus-ratio envelope of real rock
ucs = np.array([r["UCS"] for r in rows])
E = np.array([r["E"] * 1e3 for r in rows])
u = np.logspace(1, 2.6, 50)
ax2[0].fill_between(u, 200 * u, 500 * u, color=CR, alpha=.18, lw=0,
                    label="Deere & Miller: E/UCS 200–500")
for r, c in zip(rows, ["#2e8b57"] * 3 + ["#96341f"]):
    ax2[0].plot(r["UCS"], r["E"] * 1e3, "o", ms=9, color=c)
    ax2[0].annotate(r["name"], (r["UCS"], r["E"] * 1e3), textcoords="offset points",
                    xytext=(8, -3), fontsize=9)
ax2[0].set_xscale("log"); ax2[0].set_yscale("log")
ax2[0].set_xlabel("UCS (MPa), Table 1 M4"); ax2[0].set_ylabel("Young's modulus (MPa)")
ax2[0].set_title("Is the table's stiffness rock-like?", fontsize=11)
ax2[0].legend(frameon=False, fontsize=8, loc="upper left"); ax2[0].grid(alpha=.25, which="both")

# (b) Vp of the layers vs what real rock does at that density
ax2[1].fill_betweenx([0, 8], 2100, 2200, color="#96341f", alpha=.10, lw=0)
rr = np.linspace(1900, 2900, 200)
ax2[1].plot(rr, (rr / 1741.0) ** 4, lw=2, color=CR, label="Gardner (1974)")
for r, c in zip(rows, ["#2e8b57"] * 3 + ["#96341f"]):
    ax2[1].plot(r["rho"], r["Vp"] / 1000, "o", ms=9, color=c)
    dy = -16 if r["name"] == "upper" else 9
    ax2[1].annotate(r["name"], (r["rho"], r["Vp"] / 1000), textcoords="offset points",
                    xytext=(-8, dy), fontsize=9, ha="right")
ax2[1].plot([2100, 2200], [4.25, 4.25], lw=7, color="#96341f", alpha=.55, solid_capstyle="butt")
ax2[1].annotate("a real evaporite décollement\nsits here — fast, not slow",
                (2150, 4.25), xytext=(2470, 2.3), fontsize=8, color="#96341f",
                ha="center", arrowprops=dict(arrowstyle="->", color="#96341f", lw=1))
ax2[1].set_xlabel("bulk density (kg/m³)"); ax2[1].set_ylabel("Vp (km/s)")
ax2[1].set_ylim(0.5, 7); ax2[1].set_title("…and is the velocity it implies?", fontsize=11)
ax2[1].legend(frameon=False, fontsize=9); ax2[1].grid(alpha=.25)

# (c) reflection coefficients
pairs = [("upper/middle", +0.147), ("middle/lower", +0.103), ("lower/décollement", -0.691)]
cols = [CD, CD, "#96341f"]
ax2[2].barh([p[0] for p in pairs], [p[1] for p in pairs], color=cols, height=.55)
ax2[2].axvspan(-0.2, 0.2, color=CR, alpha=.16, lw=0)
ax2[2].text(0, -0.62, "|R| < 0.2: ordinary crustal interface", ha="center", fontsize=8, color=CR)
ax2[2].set_ylim(-0.9, 2.5)
for i, (_, val) in enumerate(pairs):
    ax2[2].text(val + (.04 if val > 0 else -.04), i, f"{val:+.3f}", va="center",
                ha="left" if val > 0 else "right", fontsize=9)
ax2[2].set_xlim(-0.95, 0.45); ax2[2].set_xlabel("normal-incidence reflection coefficient")
ax2[2].set_title("What the stack would reflect", fontsize=11); ax2[2].grid(alpha=.25, axis="x")
fig2.suptitle("Supplementary Table 1 audited three ways — the three crustal layers pass, "
              "the décollement fails all three", fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.92])
fig2.savefig("fig3d_paper_audit.png", dpi=140)
print("saved fig3d_paper_audit.png")

# --- Figure 3: 5 Hz sections, published vs paper ---------------------
fig3, ax3 = plt.subplots(2, 1, figsize=(15, 6.4))
for k, (key, lab) in enumerate([("published", "as published (Vp_ini 2.0/3.0/4.0 km/s)"),
                                ("paper", "Table 1 E and ρ (Vp_ini 3.96/4.90/5.58 km/s)")]):
    s = D[f"sec_{key}"]
    lim = np.nanpercentile(np.abs(s[np.isfinite(s)]), 99)
    ax3[k].imshow(np.ma.array(s.T, mask=msk | ~np.isfinite(s.T)), origin="lower",
                  extent=XZ, aspect=VE, cmap="gray", vmin=-lim, vmax=lim)
    ax3[k].set_title(f"{lab} — 5 Hz", fontsize=11)
    ax3[k].set_xlabel("X (km)"); ax3[k].set_ylabel("Z (km)")
fig3.suptitle("Synthetic section at the frequency the 250 m grid supports. The bright basal "
              "reflector in the lower panel is the décollement, R = −0.69.", fontsize=11)
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig("fig3d_paper_sections.png", dpi=140)
print("saved fig3d_paper_sections.png")
