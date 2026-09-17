"""Velocity sections and 1D-convolution synthetic seismic from a
vp_from_strain.py result.

Two products, in that order:

  1. the VELOCITY MODEL as cross-sections and map views -- Vp, impedance,
     and the part of Vp the strain is responsible for;
  2. the SYNTHETIC SEISMIC SECTION, Botter et al. (2014) step 3 in its
     lightweight form: impedance -> two-way time -> normal-incidence
     reflectivity -> Ricker convolution -> back to depth.

The wavelet frequency is tied to what the model can carry. A DEM cannot
contain structure finer than its particle spacing, so the honest upper
frequency is f = Vp / (4 * spacing), not Vp / (4 * dx) -- a fine
interpolation grid does not add information the pack does not have.

Usage:
  python3 seismic_section.py [npz] [tag] [--dx 20] [--freq 25]
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

DT = 0.001          # s, time sampling for the convolution


def ricker(f, dt, length=0.256):
    t = np.arange(-length / 2, length / 2 + dt, dt)
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def trace(dz, vp, rho, f, dt=DT):
    """One zero-offset trace down a column of uniform dz."""
    if len(vp) < 4:
        return np.full_like(vp, np.nan)
    twt = np.concatenate([[0.0], np.cumsum(2 * dz / vp[:-1])])
    nt = int(np.ceil(twt[-1] / dt)) + 1
    tu = np.arange(nt) * dt
    Z = np.interp(tu, twt, rho * vp)
    refl = np.zeros(nt)
    refl[:-1] = np.diff(Z) / (Z[:-1] + Z[1:])
    # anchor the convolution on refl: np.convolve(..., "same") returns the
    # length of the LONGER input, so a short trace under a long wavelet
    # comes back the wrong length.
    w = ricker(f, dt)
    seis = np.convolve(refl, w, mode="full")[(len(w) - 1) // 2:][:nt]
    return np.interp(twt, tu, seis)


def section(VP, RHO, live, dz, f):
    """VP/RHO are (n_trace, nz) with z increasing upward."""
    out = np.full(VP.shape, np.nan)
    for i in range(VP.shape[0]):
        col = np.where(live[i])[0]
        if len(col) < 4:
            continue
        rows = np.arange(col.max(), col.min() - 1, -1)
        out[i, rows] = trace(dz, VP[i, rows], RHO[i, rows], f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="./results/vp_model3_nocap.npz")
    ap.add_argument("tag", nargs="?", default="model3_nocap")
    ap.add_argument("--dx", type=float, default=20.0)
    ap.add_argument("--freq", type=float, default=25.0)
    a = ap.parse_args()

    D = np.load(a.npz)
    X1, Vp, rho = D["X1"], D["Vp"] * 1000.0, D["rho"]
    dVp = 100.0 * (D["Vp"] / D["Vp_ini"] - 1.0)

    spacing = np.median(cKDTree(D["X0"]).query(D["X0"], k=2)[0][:, 1])
    print(f"{len(X1)} particles, median spacing {spacing:.1f} m")
    print("resolution: the pack cannot carry structure finer than its spacing, so")
    for v, lab in ((3400.0, "shallow, 3.4 km/s"), (5900.0, "deep, 5.9 km/s")):
        print(f"  {lab:>18}: f_max = Vp / (4 x spacing) = {v/(4*spacing):5.1f} Hz")
    print(f"  using {a.freq:.0f} Hz on a {a.dx:.0f} m grid\n")

    gx, gy, gz = (np.arange(X1[:, k].min(), X1[:, k].max() + a.dx, a.dx) for k in range(3))
    tree = cKDTree(X1)

    def plane(fixed, value, u, v):
        U, V = np.meshgrid(u, v, indexing="ij")
        cols = [None] * 3
        free = [k for k in range(3) if k != fixed]
        cols[free[0]], cols[free[1]] = U.ravel(), V.ravel()
        cols[fixed] = np.full(U.size, value)
        dist, idx = tree.query(np.column_stack(cols), k=6)
        w = 1.0 / np.maximum(dist, 1e-9) ** 2
        live = (dist[:, 0] < 2 * a.dx).reshape(U.shape)
        return (lambda q: ((w * q[idx]).sum(1) / w.sum(1)).reshape(U.shape)), live

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
        "axes.edgecolor": "#555", "axes.labelcolor": "#222",
        "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold"})

    trc = np.load("./results/trace_nocap.npy")
    y_fault = float(np.interp(0.5 * (gx[0] + gx[-1]), trc[:, 0], trc[:, 1]))
    EXT_YZ = [gy.min() / 1000, gy.max() / 1000, gz.min() / 1000, gz.max() / 1000]
    EXT_XZ = [gx.min() / 1000, gx.max() / 1000, gz.min() / 1000, gz.max() / 1000]

    # ---- product 1: the velocity model -------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 7.6))
    x_mid = 0.5 * (gx[0] + gx[-1])
    g, live = plane(0, x_mid, gy, gz)
    mk = ~live.T
    VP, RHO, DV = g(Vp), g(rho), g(dVp)

    def panel(p, field, title, ext, xlab, **kw):
        im = p.imshow(np.ma.array(field.T, mask=mk), origin="lower", extent=ext,
                      aspect="equal", **kw)
        p.set_title(title, fontsize=10.5); p.set_xlabel(xlab); p.set_ylabel("Z (km)")
        plt.colorbar(im, ax=p, shrink=.86, pad=.012)
        return im

    panel(ax[0, 0], VP / 1000, f"a  Vp (km/s), across strike at X = {x_mid/1000:.2f} km",
          EXT_YZ, "Y (km)", cmap="turbo")
    panel(ax[0, 1], VP * RHO / 1e6, "b  acoustic impedance (10⁶ kg m⁻² s⁻¹)",
          EXT_YZ, "Y (km)", cmap="cividis")
    panel(ax[1, 0], DV, "c  the strain's share:  Vp / Vp_ini − 1  (%)",
          EXT_YZ, "Y (km)", cmap="RdBu_r", vmin=-15, vmax=15)
    for p in (ax[0, 0], ax[0, 1], ax[1, 0]):
        p.axvline(y_fault / 1000, color="w", ls=":", lw=1.2)

    gxz, live_xz = plane(1, y_fault, gx, gz)
    mk_xz = ~live_xz.T
    im = ax[1, 1].imshow(np.ma.array(gxz(Vp).T / 1000, mask=mk_xz), origin="lower",
                         extent=EXT_XZ, aspect="equal", cmap="turbo")
    ax[1, 1].set_title(f"d  Vp along the fault, Y = {y_fault/1000:.2f} km", fontsize=10.5)
    ax[1, 1].set_xlabel("X (km)"); ax[1, 1].set_ylabel("Z (km)")
    plt.colorbar(im, ax=ax[1, 1], shrink=.86, pad=.012)

    fig.suptitle("Velocity model from the DEM strain — cap removed, 47,802 particles",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"fig_seis_{a.tag}_velocity.png", dpi=145)
    print(f"saved fig_seis_{a.tag}_velocity.png")

    # ---- product 2: the synthetic section ----------------------------
    dz = a.dx
    S = section(VP, RHO, live, dz, a.freq)
    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 7.6))
    lim = np.nanpercentile(np.abs(S), 99)
    im = ax2[0, 0].imshow(np.ma.array(S.T, mask=mk | ~np.isfinite(S.T)), origin="lower",
                          extent=EXT_YZ, aspect="equal", cmap="gray", vmin=-lim, vmax=lim)
    ax2[0, 0].set_title(f"a  synthetic section across strike, {a.freq:.0f} Hz", fontsize=10.5)
    ax2[0, 0].set_xlabel("Y (km)"); ax2[0, 0].set_ylabel("Z (km)")
    ax2[0, 0].axvline(y_fault / 1000, color="#39d0ff", ls=":", lw=1.2)
    plt.colorbar(im, ax=ax2[0, 0], shrink=.86, pad=.012)

    S_x = section(gxz(Vp), gxz(rho), live_xz, dz, a.freq)
    lim2 = np.nanpercentile(np.abs(S_x), 99)
    im = ax2[0, 1].imshow(np.ma.array(S_x.T, mask=mk_xz | ~np.isfinite(S_x.T)), origin="lower",
                          extent=EXT_XZ, aspect="equal", cmap="gray", vmin=-lim2, vmax=lim2)
    ax2[0, 1].set_title(f"b  along the fault, Y = {y_fault/1000:.2f} km", fontsize=10.5)
    ax2[0, 1].set_xlabel("X (km)"); ax2[0, 1].set_ylabel("Z (km)")
    plt.colorbar(im, ax=ax2[0, 1], shrink=.86, pad=.012)

    # c: the only thing a strike-slip fault can do to a dip section --
    # no throw means no reflector offset, so what shows is the TIME SAG
    # that the low-velocity damage zone puts on everything beneath it.
    # measured from a flat datum, not from the ragged top of the pack:
    # without it every trace carries its own topographic static and the
    # velocity signal is buried in +/-10 ms of jitter.
    DATUM = -140.0

    def twt_to(z_target):
        out = np.full(len(gy), np.nan)
        k_t = np.argmin(np.abs(gz - z_target))
        k_d = np.argmin(np.abs(gz - DATUM))
        for i in range(len(gy)):
            col = np.where(live[i])[0]
            if len(col) < 4 or col.max() < k_d:
                continue
            rows = np.arange(k_d, max(col.min(), k_t) - 1, -1)
            if len(rows) > 1:
                out[i] = np.sum(2 * dz / VP[i, rows[:-1]])
        return out * 1000.0

    far = np.abs(gy - y_fault) > 400
    for zt, c in ((-300.0, "#15616d"), (-600.0, "#b4531b"), (-900.0, "#2e8b57")):
        t = twt_to(zt)
        ax2[1, 0].plot(gy / 1000, t - np.nanmean(t[far & np.isfinite(t)]), lw=1.9, color=c,
                       label=f"to the {zt:.0f} m level")
    ax2[1, 0].axvline(y_fault / 1000, color="#666", ls=":", lw=1.2)
    ax2[1, 0].axhline(0, color="#999", lw=.8)
    ax2[1, 0].axhspan(-1000 / a.freq / 4, 1000 / a.freq / 4, color="#888", alpha=.12, lw=0)
    ax2[1, 0].text(1.98, -1000 / a.freq / 4 * 0.75, f"± quarter period at {a.freq:.0f} Hz",
                   fontsize=8, color="#555", ha="right")
    ax2[1, 0].set_xlabel("Y (km)"); ax2[1, 0].set_ylabel("two-way time, relative (ms)")
    ax2[1, 0].set_title(f"c  the strike-slip signature: a velocity pull-down from a\n"
                        f"{DATUM:.0f} m datum — not a reflector offset", fontsize=10.5)
    ax2[1, 0].legend(frameon=False, fontsize=8.5); ax2[1, 0].grid(alpha=.22)

    Sf = section(VP, RHO, live, dz, 40.0)
    lf = np.nanpercentile(np.abs(Sf), 99)
    ax2[1, 1].imshow(np.ma.array(Sf.T, mask=mk | ~np.isfinite(Sf.T)), origin="lower",
                     extent=EXT_YZ, aspect="equal", cmap="gray", vmin=-lf, vmax=lf)
    ax2[1, 1].set_title("d  40 Hz — past what a 41 m particle spacing can carry;\n"
                        "the extra detail is interpolation", fontsize=10.5)
    ax2[1, 1].set_xlabel("Y (km)"); ax2[1, 1].set_ylabel("Z (km)")
    ax2[1, 1].axvline(y_fault / 1000, color="#39d0ff", ls=":", lw=1.2)

    fig2.suptitle("Botter et al. (2014) step 3: impedance → 1D-convolution synthetic section, "
                  "zero-phase Ricker", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(f"fig_seis_{a.tag}_section.png", dpi=145)
    print(f"saved fig_seis_{a.tag}_section.png")

    # ---- what the section actually measures --------------------------
    print("\nreflectivity budget on the across-strike section:")
    Zi = VP * RHO
    R = np.zeros_like(Zi)
    R[:, :-1] = np.diff(Zi, axis=1) / (Zi[:, :-1] + Zi[:, 1:])
    zc = 0.5 * (gz[:-1] + gz[1:])
    for zlo, zhi, lab in ((-350, -250, "the −300 m layer top"),
                          (-650, -550, "the −600 m layer top"),
                          (-1000, -100, "everything")):
        m = live[:, :-1] & (zc[None, :] >= zlo) & (zc[None, :] < zhi)
        print(f"  {lab:>22}: mean |R| {np.abs(R[:, :-1][m]).mean():.4f}   "
              f"max |R| {np.abs(R[:, :-1][m]).max():.4f}")
    near = np.abs(gy - y_fault) < 110
    far = np.abs(gy - y_fault) > 400
    print(f"\n  mean |R| within 110 m of the fault : {np.abs(R[near]).mean():.4f}")
    print(f"  mean |R| beyond 400 m              : {np.abs(R[far]).mean():.4f}")


if __name__ == "__main__":
    main()
