"""Measure the taper of a DEM wedge, and say whether it is stable.

The calibration question is not "what taper does this friction build" --
the observed geometry is the initial condition here, so the taper starts
correct by construction. It is "does the wedge KEEP that taper under the
imposed convergence". Too much basal friction and it thickens and
steepens; too little and it spreads and shallows. The basal friction at
which d(alpha)/d(shortening) is about zero is the calibrated one.

So the measurement is a difference: surface slope before convergence
against surface slope after, both taken from the exported particle
positions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nankai import geometry as G
from nankai.model_spec import KM, Model


def surface_profile(x_m, z_m, dx=500.0, min_count=8):
    """Topographic surface in the DATA frame: the shallowest particle in
    each x bin. Returns (x_km, depth_km), bins with too few particles
    dropped so the ragged ends do not bias a fit."""
    x_km, d_km = np.asarray(x_m) / KM, -np.asarray(z_m) / KM
    edges = np.arange(x_km.min(), x_km.max() + dx / KM, dx / KM)
    idx = np.clip(np.digitize(x_km, edges) - 1, 0, len(edges) - 2)
    xc, top = [], []
    for k in range(len(edges) - 1):
        s = idx == k
        if s.sum() >= min_count:
            xc.append(0.5 * (edges[k] + edges[k + 1]))
            top.append(np.percentile(d_km[s], 2.0))     # 2nd pct, not min: robust
    return np.array(xc), np.array(top)


def fit_slope(x_km, d_km, x_range):
    """Surface slope in degrees over an x window, positive where the sea
    floor deepens toward the trench (decreasing x), which is the sign
    convention of alpha."""
    s = (x_km >= x_range[0]) & (x_km <= x_range[1])
    if s.sum() < 4:
        return np.nan
    slope = np.polyfit(x_km[s], d_km[s], 1)[0]          # km of depth per km of x
    return float(np.degrees(np.arctan(-slope)))


def measure(pos_model, beta, x_range=(2.0, 29.0), dx=500.0):
    """pos_model is (n, 3) in the MODEL frame, as PFC exports it."""
    m = Model(beta=beta)
    x, z = m.to_data(pos_model[:, 0], pos_model[:, 2])
    xc, top = surface_profile(x, z, dx)
    return fit_slope(xc, top, x_range), xc, top


def target_alpha(x_range=(2.0, 29.0), n=400):
    """The digitised sea floor's slope measured THE SAME WAY as a model
    run's -- least squares over the same window. geometry.summary()
    reports a two-point chord instead, which differs by about 0.3 deg
    because the sea floor is slightly convex; comparing a chord with a
    regression is the kind of mismatch that looks like a modelling error
    and is not one."""
    x = np.linspace(*x_range, n)
    return fit_slope(x, G.depth_at("seafloor", x), x_range)


def compare(init_pos, pos, beta, x_range=(2.0, 29.0)):
    a0, _, _ = measure(init_pos, beta, x_range)
    a1, _, _ = measure(pos, beta, x_range)
    at = target_alpha(x_range)
    b = G.summary()["decollement_dip_deg"]
    return dict(alpha_initial=a0, alpha_final=a1, d_alpha=a1 - a0,
                alpha_target=at, beta=b,
                taper_final=a1 + b, taper_target=at + b)


def _self_test():
    """The preview packing is built to the digitised sea floor, so the
    measurement has to return the digitised alpha."""
    m = Model()
    px, pz = m.preview_packing()
    X, Z = m.to_model(px, pz)
    a, xc, top = measure(np.column_stack([X, np.zeros_like(X), Z]), m.beta)
    target = target_alpha()
    chord = G.summary()["surface_slope_deg"]
    print(f"self-test: alpha from the preview packing      {a:.3f} deg")
    print(f"           alpha from the digitised sea floor  {target:.3f} deg  (same fit)")
    print(f"           difference                          {a - target:+.3f} deg")
    print(f"           (geometry.summary()'s chord is      {chord:.3f} deg -- a different")
    print(f"            measure of the same surface, not a discrepancy)")
    assert abs(a - target) < 0.15, "surface fit disagrees with the digitised section"
    print("           OK")


if __name__ == "__main__":
    _self_test()
