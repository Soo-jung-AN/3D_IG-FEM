"""Digitised geometry of the NanTroSEIZE transect.

Control points read off the published seismic section (Kumano Basin ->
megasplay zone -> imbricate thrust zone -> frontal thrust zone, 0-45 km,
0-9 km depth, VE 1.5x). x = 0 at the trench, increasing landward, as in
the figure; depth positive downwards, both in km.

THESE ARE EYEBALLED FROM THE FIGURE, not from the SEGY. They are here to
be corrected: run `python3 nankai/geometry.py` to draw them at VE 1.5x,
put the result next to the original, and move the numbers until the two
agree. Everything downstream reads this file, so a correction here
propagates.
"""
import numpy as np

# ---- horizons: (x_km, depth_km) landward-increasing x ----------------
SEAFLOOR = [(0.0, 2.30), (2.0, 2.25), (5.0, 2.12), (8.0, 2.00),
            (11.0, 1.90), (14.0, 1.78), (17.0, 1.68), (20.0, 1.55),
            (23.0, 1.42), (26.0, 1.25), (28.0, 1.10), (29.5, 0.95),
            (30.2, 0.45), (31.5, 0.28), (35.0, 0.20), (40.0, 0.16),
            (45.0, 0.15)]

DECOLLEMENT = [(0.0, 4.55), (5.0, 4.75), (10.0, 5.00), (15.0, 5.25),
               (20.0, 5.55), (25.0, 5.90), (30.0, 6.30), (34.0, 6.75),
               (38.0, 7.25), (42.0, 7.75), (45.0, 8.05)]

TOP_CRUST = [(0.0, 5.45), (5.0, 5.65), (10.0, 5.90), (15.0, 6.20),
             (20.0, 6.50), (25.0, 6.85), (30.0, 7.30), (34.0, 7.75),
             (38.0, 8.25), (42.0, 8.75), (45.0, 9.00)]

# the megasplay: listric, rooting into the plate interface landward and
# cutting the sea floor in the middle of the section
MEGASPLAY = [(38.0, 7.20), (35.0, 5.90), (33.0, 4.80), (31.0, 3.70),
             (29.0, 2.60), (27.5, 1.80), (26.5, 1.22)]

# base of the Kumano Basin fill, i.e. top of the inner accretionary prism
KUMANO_BASE = [(29.8, 0.95), (31.0, 1.35), (33.0, 1.60), (36.0, 1.75),
               (40.0, 1.85), (45.0, 1.95)]

ZONES = [("Frontal thrust zone", 0.0, 8.0),
         ("Imbricate thrust zone", 8.0, 22.0),
         ("Megasplay zone", 22.0, 31.0),
         ("Kumano Basin", 31.0, 45.0)]

VE = 1.5          # vertical exaggeration of the published figure


def horizon(name):
    return np.array({"seafloor": SEAFLOOR, "decollement": DECOLLEMENT,
                     "top_crust": TOP_CRUST, "megasplay": MEGASPLAY,
                     "kumano_base": KUMANO_BASE}[name], dtype=float)


def depth_at(name, x_km):
    """Depth of a horizon, km, at one or many x. Megasplay excluded: it
    is multivalued in x only over its own range, so use horizon()."""
    h = horizon(name)
    return np.interp(x_km, h[:, 0], h[:, 1])


def dips(name):
    """True dip in degrees along a horizon, i.e. with the figure's 1.5x
    vertical exaggeration removed."""
    h = horizon(name)
    d = np.diff(h, axis=0)
    # magnitude only: the megasplay's control points run landward-to-
    # trenchward, so a signed arctan2 would report its dip as ~-150 deg.
    return (np.degrees(np.arctan(np.abs(d[:, 1] / d[:, 0]))),
            0.5 * (h[:-1, 0] + h[1:, 0]))


def summary():
    sf, dc = horizon("seafloor"), horizon("decollement")
    a = np.degrees(np.arctan((sf[0, 1] - depth_at("seafloor", 29.5)) / 29.5))
    b = np.degrees(np.arctan((depth_at("decollement", 30) - dc[0, 1]) / 30.0))
    ms, _ = dips("megasplay")
    return dict(surface_slope_deg=a, decollement_dip_deg=b, taper_deg=a + b,
                megasplay_dip_deg=(ms.min(), ms.max()),
                prism_thickness_km=(dc[0, 1] - sf[0, 1],
                                    depth_at("decollement", 45) - depth_at("seafloor", 45)))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary()
    print("As digitised (true angles, VE removed):")
    print(f"  surface slope alpha   {s['surface_slope_deg']:5.2f} deg  (outer wedge, 0-30 km)")
    print(f"  decollement dip beta  {s['decollement_dip_deg']:5.2f} deg")
    print(f"  taper alpha + beta    {s['taper_deg']:5.2f} deg")
    print(f"  megasplay dip         {s['megasplay_dip_deg'][0]:5.1f} to "
          f"{s['megasplay_dip_deg'][1]:5.1f} deg")
    print(f"  prism thickness       {s['prism_thickness_km'][0]:5.2f} km at the toe, "
          f"{s['prism_thickness_km'][1]:5.2f} km under Kumano")

    fig, ax = plt.subplots(figsize=(13, 4.2))
    for name, c, lw, lab in (("seafloor", "#1f6fb4", 2.0, "sea floor"),
                             ("kumano_base", "#7a4fa3", 1.6, "base of Kumano Basin fill"),
                             ("decollement", "#16191c", 2.2, "décollement"),
                             ("top_crust", "#8a6410", 1.8, "top of igneous crust"),
                             ("megasplay", "#96341f", 2.6, "megasplay fault")):
        h = horizon(name)
        ax.plot(h[:, 0], h[:, 1], color=c, lw=lw, label=lab)
    sf, dc = horizon("seafloor"), horizon("decollement")
    xx = np.linspace(0, 45, 400)
    ax.fill_between(xx, depth_at("seafloor", xx), depth_at("decollement", xx),
                    color="#d9c9a3", alpha=.45, lw=0)
    ax.fill_between(xx, depth_at("decollement", xx), depth_at("top_crust", xx),
                    color="#b8a97f", alpha=.55, lw=0)
    for lab, x0, x1 in ZONES:
        ax.axvline(x1, color="#888", lw=.8, ls=":")
        ax.text(0.5 * (x0 + x1), -0.45, lab, ha="center", fontsize=8.5, color="#444")
    ax.set_xlim(45, 0); ax.set_ylim(9.2, -0.8)
    ax.set_aspect(VE)
    ax.set_xlabel("distance from the trench (km)"); ax.set_ylabel("depth (km)")
    ax.set_title(f"Digitised NanTroSEIZE transect, drawn at the figure's VE = {VE}× "
                 "— compare against the original", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig("nankai/fig_geometry_check.png", dpi=150)
    print("\nsaved nankai/fig_geometry_check.png")
