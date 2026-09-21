"""Does the fault show up the way it would in real 3D seismic?

A dip section is the worst place to look for a strike-slip fault, and the
synthetic section confirms it: no throw, so no reflector offset. In
practice strike-slip faults are found on 3D coherence (semblance) time
slices, where they appear as linear discontinuities in map view.

This builds the full synthetic volume -- one trace per (x, y) column --
and computes a semblance attribute on depth slices, so the model can be
judged the way a real survey would be.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from seismic_section import trace

DX, FREQ = 20.0, 25.0
WIN = 3          # vertical half-window, samples
DEPTHS = (-200.0, -350.0, -650.0)

D = np.load("./results/vp_model3_nocap.npz")
X1, Vp, rho = D["X1"], D["Vp"] * 1000.0, D["rho"]
dVp = 100.0 * (D["Vp"] / D["Vp_ini"] - 1.0)

gx, gy, gz = (np.arange(X1[:, k].min(), X1[:, k].max() + DX, DX) for k in range(3))
GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
nodes = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
tree = cKDTree(X1)
dist, idx = tree.query(nodes, k=6)
w = 1.0 / np.maximum(dist, 1e-9) ** 2
shape = GX.shape
grid = lambda v: ((w * v[idx]).sum(1) / w.sum(1)).reshape(shape)
live = (dist[:, 0] < 2 * DX).reshape(shape)
VP, RHO, DV = grid(Vp), grid(rho), grid(dVp)
print(f"volume {shape[0]} x {shape[1]} x {shape[2]}, {live.sum():,} live cells", flush=True)

S = np.full(shape, np.nan)
for i in range(shape[0]):
    for j in range(shape[1]):
        col = np.where(live[i, j])[0]
        if len(col) < 6:
            continue
        rows = np.arange(col.max(), col.min() - 1, -1)
        S[i, j, rows] = trace(DX, VP[i, j, rows], RHO[i, j, rows], FREQ)
print("synthetic volume done", flush=True)


def semblance(vol, half=1, vwin=WIN):
    """Classic 3-by-3 trace semblance in a short vertical window."""
    v = np.nan_to_num(vol)
    n = 0
    ssum = np.zeros_like(v)
    sq = np.zeros_like(v)
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            sh = np.roll(np.roll(v, di, axis=0), dj, axis=1)
            ssum += sh
            sq += sh ** 2
            n += 1
    k = np.ones(2 * vwin + 1)
    num = np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 2, ssum ** 2)
    den = n * np.apply_along_axis(lambda a: np.convolve(a, k, "same"), 2, sq)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-30)
    return np.clip(out, 0, 1)


C = semblance(S)
print("semblance done", flush=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold"})
EXT = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]
trc = np.load("./results/trace_nocap.npy")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for j, zc in enumerate(DEPTHS):
    k = np.argmin(np.abs(gz - zc))
    m = ~live[:, :, k]
    im = ax[0, j].imshow(np.ma.array(C[:, :, k].T, mask=m.T), origin="lower", extent=EXT,
                         aspect="equal", cmap="gray_r", vmin=np.nanpercentile(C[:, :, k], 2),
                         vmax=1.0)
    ax[0, j].set_title(f"{'abc'[j]}  semblance at Z = {zc:.0f} m", fontsize=10.5)
    ax[0, j].set_xlabel("X (km)"); ax[0, j].set_ylabel("Y (km)")
    plt.colorbar(im, ax=ax[0, j], shrink=.86, pad=.012)

    im2 = ax[1, j].imshow(np.ma.array(DV[:, :, k].T, mask=m.T), origin="lower", extent=EXT,
                          aspect="equal", cmap="RdBu_r", vmin=-15, vmax=15)
    ax[1, j].plot(trc[:, 0] / 1000, trc[:, 1] / 1000, ":", lw=1.3, color="#111")
    ax[1, j].set_title(f"{'def'[j]}  ΔVp from strain at the same depth  (%)", fontsize=10.5)
    ax[1, j].set_xlabel("X (km)"); ax[1, j].set_ylabel("Y (km)")
    plt.colorbar(im2, ax=ax[1, j], shrink=.86, pad=.012)

fig.suptitle("How a real survey would look for this fault: semblance on depth slices of the "
             "synthetic volume (top), against the velocity anomaly that produced it (bottom)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_coherence_model3.png", dpi=145)
print("saved fig_coherence_model3.png")

ytr = np.interp(gx, trc[:, 0], trc[:, 1])
print("\nsemblance contrast, fault vs background:")
for zc in DEPTHS:
    k = np.argmin(np.abs(gz - zc))
    c = C[:, :, k]
    d = np.abs(gy[None, :] - ytr[:, None])
    ok = live[:, :, k]
    near = ok & (d < 110); far = ok & (d > 400)
    print(f"  Z = {zc:6.0f} m:  fault {c[near].mean():.4f}   background {c[far].mean():.4f}"
          f"   drop {100*(1-c[near].mean()/c[far].mean()):5.2f}%"
          f"   background sd {c[far].std():.4f}"
          f"   -> {(c[far].mean()-c[near].mean())/c[far].std():5.2f} sigma")
