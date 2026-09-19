"""Everything about the Nankai DEM model that does not need PFC.

Deliberately free of `itasca`, so it can be tested, plotted and argued
with outside PFC. build_model.py is the thin PFC-side wrapper that calls
into this.

WHY GRAVITY IS TILTED. The décollement dips landward by beta. Rather
than build a dipping basal boundary, the whole section is rotated by
+beta so the décollement is horizontal, and gravity is rotated with it.
That is not a shortcut: critical-taper theory (Davis, Suppe & Dahlen,
1983) is normally derived in exactly this frame. In it,

    g = (g sin beta, 0, -g cos beta)

The +x component is landward, which is down-dip on the décollement -- the
plate goes down as you go landward -- so a block on the flattened base
slides landward, as it should. The trenchward push comes from the
surface, which in this frame slopes down toward the trench at the full
taper alpha + beta.

The décollement is not planar; it steepens landward from about 3 to 5.6
degrees. Rotating by its mean leaves a gentle residual curvature, which
is real and is kept.
"""
import os
import sys

import numpy as np

# importable as `nankai.model_spec` and runnable as `python3
# nankai/model_spec.py`, which PFC's interpreter will want to do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nankai import geometry as G

KM = 1000.0

# unit codes
UNITS = {"kumano": 1, "inner_prism": 2, "outer_prism": 3,
         "decollement": 4, "underthrust": 5, "crust": 6}
UNIT_NAMES = {v: k for k, v in UNITS.items()}

# Per-unit properties. Densities are bulk values for water-saturated
# marine sediment and oceanic crust; E and the bond strengths are
# starting guesses in the style of An & So (2026) Table 1 but an order of
# magnitude weaker, because this is unlithified to poorly lithified
# sediment rather than crystalline crust. THEY ARE NOT CALIBRATED --
# basal friction in particular is the knob that sets the taper, and it
# has to be swept (see taper_target).
PROPERTIES = {
    "kumano":      dict(density=1900.0, E=1.0e9, friction=0.45, tensile=0.3e6, cohesion=1.5e6),
    "inner_prism": dict(density=2300.0, E=8.0e9, friction=0.55, tensile=3.0e6, cohesion=15.0e6),
    "outer_prism": dict(density=2100.0, E=3.0e9, friction=0.50, tensile=1.0e6, cohesion=5.0e6),
    "decollement": dict(density=2050.0, E=1.5e9, friction=0.15, tensile=0.05e6, cohesion=0.25e6),
    "underthrust": dict(density=2150.0, E=2.0e9, friction=0.45, tensile=0.8e6, cohesion=4.0e6),
    "crust":       dict(density=2700.0, E=50.0e9, friction=0.60, tensile=10.0e6, cohesion=50.0e6),
}

# Seeded faults: the mapped structures, given as weak zones so the model
# is asked which of them takes up the current increment of slip rather
# than being asked to grow them from nothing. WEAK_FACTOR multiplies the
# host unit's bond strengths inside the zone.
WEAK_FACTOR = 0.15
MEGASPLAY_HALF_WIDTH = 150.0      # m
THRUST_HALF_WIDTH = 100.0         # m

# Imbricate and frontal thrusts, as (x_top_km, dip_deg): each cuts up
# from the décollement to the sea floor, dipping landward, which is the
# vergence of an accretionary prism. Steeper and more closely spaced at
# the toe, shallowing landward as earlier thrusts are carried back and
# rotated. Approximate, like geometry.py, and meant to be corrected
# against the section.
THRUSTS = [(1.0, 38.0), (2.5, 36.0), (4.0, 34.0), (6.0, 32.0),
           (8.5, 30.0), (11.5, 28.0), (15.0, 27.0), (19.0, 26.0),
           (22.5, 25.0)]

DECOLLEMENT_THICKNESS = 200.0     # m, the weak basal layer
CONVEYOR_THICKNESS = 300.0        # m of velocity-controlled particles


def mean_dip():
    """Mean décollement dip over the section, degrees."""
    d = G.horizon("decollement")
    return float(np.degrees(np.arctan((d[-1, 1] - d[0, 1]) / (d[-1, 0] - d[0, 0]))))


class Model:
    def __init__(self, r_mean=33.0, r_ratio=1.5, slab=264.0, beta=None,
                 x_range=(0.0, 45.0)):
        self.r_mean, self.r_ratio, self.slab = r_mean, r_ratio, slab
        self.beta = mean_dip() if beta is None else beta
        self.x0, self.x1 = x_range

    # ---- frames ------------------------------------------------------
    def to_model(self, x_m, z_m):
        """Data frame (x landward, z up, metres) -> model frame, rotated
        by +beta so the décollement is horizontal."""
        c, s = np.cos(np.radians(self.beta)), np.sin(np.radians(self.beta))
        return x_m * c - z_m * s, x_m * s + z_m * c

    def to_data(self, X, Z):
        c, s = np.cos(np.radians(self.beta)), np.sin(np.radians(self.beta))
        return X * c + Z * s, -X * s + Z * c

    def gravity(self, g=9.81):
        b = np.radians(self.beta)
        return (g * np.sin(b), 0.0, -g * np.cos(b))

    # ---- geometry in the data frame ----------------------------------
    def _d(self, name, x_km):
        return G.depth_at(name, x_km)

    def envelope(self, x_km):
        """(top, bottom) depth in km of everything the model contains:
        sea floor down to the top of the igneous crust."""
        return self._d("seafloor", x_km), self._d("top_crust", x_km)

    def inside(self, x_m, z_m):
        x_km, d_km = x_m / KM, -z_m / KM
        top, bot = self.envelope(x_km)
        return (d_km >= top) & (d_km <= bot) & (x_km >= self.x0) & (x_km <= self.x1)

    def unit(self, x_m, z_m):
        """Unit code at each (x, z) in the DATA frame."""
        x_km, d_km = np.asarray(x_m) / KM, -np.asarray(z_m) / KM
        dec = self._d("decollement", x_km)
        out = np.full(x_km.shape, UNITS["outer_prism"], dtype=np.int8)

        out[d_km > self._d("top_crust", x_km) - 1e-9] = UNITS["crust"]
        below = d_km > dec
        out[below] = UNITS["underthrust"]
        out[d_km > self._d("top_crust", x_km)] = UNITS["crust"]

        band = np.abs(d_km - dec) * KM <= DECOLLEMENT_THICKNESS / 2
        out[band] = UNITS["decollement"]

        landward = x_km >= G.horizon("megasplay")[-1, 0]
        above_dec = d_km < dec
        out[landward & above_dec & (out == UNITS["outer_prism"])] = UNITS["inner_prism"]

        kb = np.interp(x_km, *G.horizon("kumano_base").T)
        in_basin = (x_km >= G.horizon("kumano_base")[0, 0]) & (d_km < kb)
        out[in_basin] = UNITS["kumano"]
        return out

    # ---- seeded faults -----------------------------------------------
    @staticmethod
    def _dist_to_polyline(x_m, z_m, poly_km):
        """Shortest distance in m from each point to a polyline given as
        (x_km, depth_km)."""
        p = np.column_stack([np.asarray(x_m).ravel(), np.asarray(z_m).ravel()])
        v = np.column_stack([poly_km[:, 0] * KM, -poly_km[:, 1] * KM])
        best = np.full(len(p), np.inf)
        for a, b in zip(v[:-1], v[1:]):
            ab = b - a
            t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
            best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
        return best.reshape(np.asarray(x_m).shape)

    def thrust_polyline(self, x_top_km, dip_deg):
        """A thrust cutting up from the décollement to the sea floor."""
        xs, ds = [], []
        x = x_top_km
        d = self._d("seafloor", x)
        step = 0.05
        while d < self._d("decollement", x) and x < self.x1:
            xs.append(x); ds.append(d)
            x += step
            d += step * np.tan(np.radians(dip_deg))
        xs.append(x); ds.append(self._d("decollement", x))
        return np.column_stack([xs, ds])

    def weak(self, x_m, z_m):
        """True where a mapped fault has been seeded."""
        w = self._dist_to_polyline(x_m, z_m, G.horizon("megasplay")) <= MEGASPLAY_HALF_WIDTH
        for x_top, dip in THRUSTS:
            w |= self._dist_to_polyline(x_m, z_m,
                                        self.thrust_polyline(x_top, dip)) <= THRUST_HALF_WIDTH
        return w

    def conveyor(self, x_m, z_m):
        """The velocity-controlled basal layer: the bottom of the
        underthrust section, which carries sediment under the wedge."""
        x_km, d_km = np.asarray(x_m) / KM, -np.asarray(z_m) / KM
        tc = self._d("top_crust", x_km)
        return (d_km <= tc) & (d_km >= tc - CONVEYOR_THICKNESS / KM)

    # ---- packing ------------------------------------------------------
    def n_particles(self, packing_fraction=0.6):
        x = np.linspace(self.x0, self.x1, 400)
        top, bot = self.envelope(x)
        # np.trapezoid is numpy >= 2; PFC 6 embeds numpy 1.13, which has
        # only np.trapz. model_spec runs inside PFC, so take whichever.
        integrate = getattr(np, "trapezoid", None) or np.trapz
        area = integrate(bot - top, x) * KM ** 2
        vol = area * self.slab
        return int(packing_fraction * vol / (4 / 3 * np.pi * self.r_mean ** 3))

    def preview_packing(self, seed=0, n_y=1):
        """A jittered lattice inside the envelope, for checking the unit
        and weak-zone assignment without PFC. Not a DEM packing -- PFC's
        ball distribute makes the real one."""
        rng = np.random.default_rng(seed)
        step = 2 * self.r_mean
        xs = np.arange(self.x0 * KM, self.x1 * KM, step)
        zt = -self._d("seafloor", xs / KM) * KM
        zb = -self._d("top_crust", xs / KM) * KM
        px, pz = [], []
        for i, x in enumerate(xs):
            z = np.arange(zb[i], zt[i], step * np.sqrt(3) / 2)
            off = (step / 2) if (i % 2) else 0.0
            px.append(np.full(len(z), x + off)); pz.append(z)
        px, pz = np.concatenate(px), np.concatenate(pz)
        j = rng.uniform(-0.15, 0.15, (2, len(px))) * step
        px, pz = px + j[0], pz + j[1]
        keep = self.inside(px, pz)
        return px[keep], pz[keep]


def taper_target():
    """What the run has to reproduce, from the digitised section."""
    s = G.summary()
    return dict(alpha=s["surface_slope_deg"], beta=s["decollement_dip_deg"],
                taper=s["taper_deg"])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    m = Model()
    px, pz = m.preview_packing()
    u, w, c = m.unit(px, pz), m.weak(px, pz), m.conveyor(px, pz)
    tt = taper_target()

    print(f"beta (mean décollement dip)  {m.beta:.2f} deg")
    print(f"gravity in the model frame   ({m.gravity()[0]:+.3f}, 0, {m.gravity()[2]:.3f}) m/s2")
    print(f"  the +x component is landward, i.e. down-dip on the flattened base")
    print(f"particles at r = {m.r_mean:.0f} m, slab {m.slab:.0f} m: {m.n_particles():,}")
    print(f"taper to reproduce: alpha {tt['alpha']:.2f} + beta {tt['beta']:.2f} "
          f"= {tt['taper']:.2f} deg")
    print(f"\npreview packing {len(px):,} points")
    for code in sorted(set(u.tolist())):
        print(f"  {UNIT_NAMES[code]:<12} {(u == code).sum():7,}  {100*(u == code).mean():5.1f}%")
    print(f"  {'seeded faults':<12} {w.sum():7,}  {100*w.mean():5.1f}%")
    print(f"  {'conveyor':<12} {c.sum():7,}  {100*c.mean():5.1f}%")

    colours = {"kumano": "#f0dfa4", "inner_prism": "#8c6f4a", "outer_prism": "#d3bb8c",
               "decollement": "#2f6f9e", "underthrust": "#8fae9a", "crust": "#5c4d3f"}
    fig, ax = plt.subplots(2, 1, figsize=(15, 8.2))

    for name, code in UNITS.items():
        s = u == code
        if s.any():
            ax[0].scatter(px[s] / KM, -pz[s] / KM, s=.6, c=colours[name], lw=0, label=name)
    ax[0].scatter(px[w] / KM, -pz[w] / KM, s=.9, c="#96341f", lw=0, label="seeded fault")
    ax[0].scatter(px[c] / KM, -pz[c] / KM, s=.9, c="#15616d", lw=0, label="conveyor")
    ax[0].set_xlim(45, 0); ax[0].set_ylim(9.2, -0.4); ax[0].set_aspect(G.VE)
    ax[0].set_xlabel("distance from the trench (km)"); ax[0].set_ylabel("depth (km)")
    ax[0].set_title(f"a  units and seeded structures, data frame, VE {G.VE}× "
                    f"— compare against the section", fontsize=11)
    ax[0].set_ylim(9.2, -0.4)
    ax[0].legend(frameon=False, fontsize=8, ncol=7, markerscale=9,
                 loc="upper center", bbox_to_anchor=(0.5, -0.20))

    MX, MZ = m.to_model(px, pz)
    ax[1].scatter(MX / KM, MZ / KM, s=.6, c=[colours[UNIT_NAMES[q]] for q in u], lw=0)
    ax[1].scatter(MX[w] / KM, MZ[w] / KM, s=.9, c="#96341f", lw=0)
    ax[1].scatter(MX[c] / KM, MZ[c] / KM, s=.9, c="#15616d", lw=0)
    gx, _, gz = m.gravity()
    x0, z0 = 6.0, -7.4
    ax[1].arrow(x0, z0, gx / 3.0, gz / 3.0, head_width=.22, color="#16191c", lw=1.6,
                length_includes_head=True)
    ax[1].text(x0 + .4, z0 - 1.5, f"g, tilted {m.beta:.1f}°", fontsize=9)
    ax[1].set_aspect(1.0)
    ax[1].set_xlabel("X (km)"); ax[1].set_ylabel("Z (km)")
    ax[1].set_title("b  the model frame the DEM actually runs in: décollement rotated flat, "
                    "gravity rotated with it, true scale", fontsize=11)
    fig.tight_layout()
    fig.savefig("nankai/fig_model_setup.png", dpi=145)
    print("\nsaved nankai/fig_model_setup.png")
