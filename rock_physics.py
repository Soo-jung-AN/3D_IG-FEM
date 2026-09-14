"""
Rock-physics relations linking DEM finite volumetric strain to synthetic
seismic velocities, following the empirical workflow of:

Botter, C., Cardozo, N., Hardy, S., Lecomte, I., Escalona, A. (2014).
From mechanical modeling to seismic imaging of faults: A synthetic
workflow to study the impact of faults on seismic reflection data.
Marine and Petroleum Geology, 57, 187-207. Equations (1)-(4), Section 2.2.
https://doi.org/10.1016/j.marpetgeo.2014.05.013

All velocities are in km/s (Han's relation, Eq. 4, is calibrated in km/s).
"""
import numpy as np

RHO_WATER = 1000.0  # kg/m3, saturated pore fluid density (Botter et al., 2014, Table 3)


def porosity_from_strain(vol_strain, phi_ini):
    """Eq. (1): linear porosity change with volumetric strain (dilatation)."""
    ev = np.clip(vol_strain, -1.0, 1.0)
    return phi_ini * (0.25 * ev + 1.0)


def density_from_porosity(phi, rho_grain, rho_fluid=RHO_WATER):
    """Eq. (2): saturated bulk density from porosity (linear mixing)."""
    return rho_grain * (1.0 - phi) + rho_fluid * phi


def vp_from_strain(vol_strain, Vp_ini):
    """Eq. (3): piecewise-quadratic (sigmoidal) Vp vs. volumetric strain,
    bounded to +/-25% change from Vp_ini at |vol_strain| = 1."""
    ev = np.clip(vol_strain, -1.0, 1.0)
    compaction = ev < 0.0
    factor = np.where(
        compaction,
        -0.25 * ev**2 - 0.5 * ev + 1.0,
        0.25 * ev**2 - 0.5 * ev + 1.0,
    )
    return Vp_ini * factor


def vs_from_vp(Vp):
    """Eq. (4): Han's (1986) empirical Vp-Vs relation (km/s), as used in
    Botter et al. (2014)."""
    return 0.794 * Vp - 0.787


def synthesize_vpvs(vol_strain, phi_ini, rho_grain, Vp_ini, rho_fluid=RHO_WATER):
    """Apply Eqs. (1)-(4) of Botter et al. (2014) to DEM finite volumetric
    strain (vol_strain = det(F) - 1), returning (phi, rho, Vp, Vs, Vp/Vs)."""
    phi = porosity_from_strain(vol_strain, phi_ini)
    rho = density_from_porosity(phi, rho_grain, rho_fluid)
    Vp = vp_from_strain(vol_strain, Vp_ini)
    Vs = vs_from_vp(Vp)
    return phi, rho, Vp, Vs, Vp / Vs


# Reference (undeformed) properties of the 3D model, by depth zone. These
# are the values main.py has always used; they live here so that anything
# else applying Eqs. (1)-(4) to this model -- compare_3d.py in particular --
# is guaranteed to use exactly the same reference state, which is the only
# way a comparison between two strain estimators is about the strain.
# They are example values and should be calibrated to the actual DEM
# materials, the way Botter et al. calibrated theirs (their Table 3).
ZONES = (
    # (z_lo, z_hi, x_lo, x_hi, rho_grain, phi_ini, Vp_ini[km/s])
    (-15e3, -11e3, None,  None,  2700.0, 0.10, 4.0),
    (-11e3,  -7e3, None,  None,  2500.0, 0.15, 3.0),
    ( -7e3,   0.0, None,  None,  2300.0, 0.25, 2.0),
    (-15e3, -13e3, 30e3,  120e3, 2100.0, 0.35, 1.5),   # overrides zone 1
)


def zoned_initial_properties(pos):
    """Reference phi, grain density and Vp for every particle, from its
    UNDEFORMED position (x, y, z). Returns (phi_ini, rho_grain, Vp_ini)."""
    x, z = pos[:, 0], pos[:, 2]
    n = len(pos)
    phi_ini = np.zeros(n)
    rho_grain = np.zeros(n)
    Vp_ini = np.zeros(n)
    for z_lo, z_hi, x_lo, x_hi, rho_g, phi0, vp0 in ZONES:
        sel = (z >= z_lo) & (z < z_hi)
        if x_lo is not None:
            sel &= (x >= x_lo) & (x <= x_hi)
        rho_grain[sel], phi_ini[sel], Vp_ini[sel] = rho_g, phi0, vp0
    return phi_ini, rho_grain, Vp_ini


# ---------------------------------------------------------------------
# Crustal calibration (Brocher, 2005, BSSA 95, 2081-2092)
#
# Han's (1986) Vp-Vs line in Eq. (4) is a water-saturated SANDSTONE fit
# over roughly 3.0-5.5 km/s. Extrapolated below ~2.5 km/s it returns Vs
# near zero and Vp/Vs above 4, i.e. Poisson's ratio approaching 0.5 for
# rock that is supposed to be consolidated. Brocher's regression fit and
# the Nafe-Drake curve are the standard crustal-scale replacements and
# stay physical over 1.5 < Vp < 8.5 km/s, which is the range a 15 km
# column actually spans.
# ---------------------------------------------------------------------

def vs_from_vp_brocher(Vp):
    """Brocher (2005) Eq. (1), the 'regression fit'. Vp, Vs in km/s;
    valid for 1.5 < Vp < 8.0 km/s."""
    return (0.7858 - 1.2344 * Vp + 0.7949 * Vp ** 2
            - 0.1238 * Vp ** 3 + 0.0064 * Vp ** 4)


def density_from_vp_nafe_drake(Vp):
    """Brocher (2005) Eq. (2), the Nafe-Drake curve. Vp in km/s, bulk
    density returned in kg/m3; valid for 1.5 < Vp < 8.5 km/s."""
    return 1000.0 * (1.6612 * Vp - 0.4721 * Vp ** 2 + 0.0671 * Vp ** 3
                     - 0.0043 * Vp ** 4 + 0.000106 * Vp ** 5)


def vp_from_density_nafe_drake(rho, lo=1.5, hi=8.5, iters=60):
    """Invert the Nafe-Drake curve: the Vp a given bulk density implies.
    The curve is monotonic over its stated range, so a bisection is exact
    to machine precision and needs no fitted inverse."""
    rho = np.asarray(rho, dtype=float)
    a = np.full(rho.shape, lo)
    b = np.full(rho.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_slow = density_from_vp_nafe_drake(m) < rho
        a = np.where(too_slow, m, a)
        b = np.where(too_slow, b, m)
    return 0.5 * (a + b)


def grain_density_for_bulk(rho_bulk, phi, rho_fluid=RHO_WATER):
    """The grain density that makes Eq. (2) reproduce a target UNDEFORMED
    bulk density at porosity phi. Without this the workflow silently
    lowers the model's density: ZONES gives Eq. (2) the DEM's own bulk
    densities as GRAIN densities, and mixing water into them puts the
    synthesized bulk 9-14% below the density the DEM used to compute its
    own lithostatic stress."""
    return (np.asarray(rho_bulk, dtype=float) - rho_fluid * phi) / (1.0 - phi)


# Bulk densities the DEM itself uses for lithostatic stress in main.py,
# with crustal porosities and the Vp each density implies through the
# Nafe-Drake curve. Same geometry as ZONES; only the properties differ.
ZONES_CRUSTAL_RHO_PHI = (
    # (z_lo, z_hi, x_lo, x_hi, rho_bulk, phi_ini)
    (-15e3, -11e3, None,  None,  2700.0, 0.01),
    (-11e3,  -7e3, None,  None,  2500.0, 0.03),
    ( -7e3,   0.0, None,  None,  2300.0, 0.10),
    (-15e3, -13e3, 30e3,  120e3, 2100.0, 0.05),   # weak seed layer
)


def crustal_initial_properties(pos):
    """Reference properties tied to the DEM's own density structure:
    Vp_ini from the Nafe-Drake curve, and a grain density chosen so the
    undeformed bulk density is the one the DEM assumed. Returns
    (phi_ini, rho_grain, Vp_ini) with the same signature as
    zoned_initial_properties, so it drops straight into synthesize_vpvs."""
    x, z = pos[:, 0], pos[:, 2]
    n = len(pos)
    phi_ini = np.zeros(n)
    rho_bulk = np.zeros(n)
    for z_lo, z_hi, x_lo, x_hi, rho_b, phi0 in ZONES_CRUSTAL_RHO_PHI:
        sel = (z >= z_lo) & (z < z_hi)
        if x_lo is not None:
            sel &= (x >= x_lo) & (x <= x_hi)
        rho_bulk[sel], phi_ini[sel] = rho_b, phi0
    Vp_ini = vp_from_density_nafe_drake(rho_bulk)
    return phi_ini, grain_density_for_bulk(rho_bulk, phi_ini), Vp_ini


# ---------------------------------------------------------------------
# The DEM's own elastic constants.
#
# An & So (2026), Communications Earth & Environment, Supplementary
# Table 1 ("Numerical particle properties for the five DE models M1-M5")
# gives density, Young's modulus and friction angle per layer, plus
# per-model strengths. The seismic properties of this model therefore do
# not need an empirical Vp relation at all: E and rho give Vp and Vs
# directly, and the only free parameter left is Poisson's ratio, which
# the table does not list.
#
# E is listed among "particle properties", i.e. it is the contact-law
# modulus. For a dense 3D bonded pack the emergent macroscopic modulus is
# close to it (Potyondy & Cundall, 2004, found E_macro ~ 1.1 E_c), and
# the table's own E/UCS ratios land at 355 / 287 / 191 for the three
# crustal layers -- squarely in Deere & Miller's medium-to-high modulus
# ratio range for real rock -- so E is used here as the macroscopic
# modulus. E_macro = 0.7 E_c instead would lower every velocity by 16%.
# ---------------------------------------------------------------------

NU_DEM = 0.25        # not in the table; 0.25 is the Poisson-solid value

ZONES_PAPER = (
    # (z_lo, z_hi, x_lo, x_hi, rho, E[GPa], friction[deg], UCS_M4[MPa], name)
    (-15e3, -11e3, None,  None,  2700.0, 70.0, 30.0, 197.13, "lower"),
    (-11e3,  -7e3, None,  None,  2500.0, 50.0, 30.0, 174.29, "middle"),
    ( -7e3,   0.0, None,  None,  2300.0, 30.0, 30.0, 156.94, "upper"),
    (-15e3, -13e3, 30e3,  120e3, 2100.0,  3.0, 10.0,  90.90, "decollement"),
)


def vp_vs_from_E_rho(E, rho, nu=NU_DEM):
    """Isotropic elastic velocities from Young's modulus and density.
    E in Pa, rho in kg/m3, velocities in m/s. Vp/Vs depends only on nu:
    sqrt(2(1-nu)/(1-2nu)), i.e. 1.732 at nu = 0.25."""
    M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))   # P-wave modulus
    G = E / (2.0 * (1.0 + nu))
    return np.sqrt(M / rho), np.sqrt(G / rho)


def paper_layer_fields(pos):
    """Per-particle bulk density, Young's modulus and layer index from
    Supplementary Table 1, assigned on the UNDEFORMED position."""
    x, z = pos[:, 0], pos[:, 2]
    n = len(pos)
    rho = np.zeros(n)
    E = np.zeros(n)
    layer = np.full(n, -1)
    for k, (z_lo, z_hi, x_lo, x_hi, rho_b, E_gpa, _, _, _) in enumerate(ZONES_PAPER):
        sel = (z >= z_lo) & (z < z_hi)
        if x_lo is not None:
            sel &= (x >= x_lo) & (x <= x_hi)
        rho[sel], E[sel], layer[sel] = rho_b, E_gpa * 1e9, k
    return rho, E, layer


def density_from_mass_conservation(rho_ini, vol_strain,
                                   max_compaction=0.03, max_dilatation=0.25):
    """rho = rho_ini / det(F).

    For a low-porosity crust this replaces Botter Eq. (2). Eq. (2) routes
    the density change through a porosity change, which is the right
    physics for a porous sandstone but does nothing at crustal porosity.
    det(F) is by definition the volume ratio of the same material, so
    dividing by it is exact for the DEM's own kinematics and needs no
    porosity at all.

    The volume ratio is bounded first, and ASYMMETRICALLY, which is the
    part that matters for a crustal model. A rock can only densify by
    closing the porosity it has -- 1-3% in crystalline crust, hence
    max_compaction -- but it can dilate a long way further by fracturing,
    which is exactly what a fault damage zone does. Unbounded, the DEM's
    own outliers (det(F) - 1 down to -0.99 on residual slivers) would
    produce 100x densities; bounded symmetrically at Botter's +/-25%, the
    median compaction of -0.07 would still imply a 7% densification of
    crystalline rock, which has no pore space to give.

    This bound is where the Botter workflow strains against a crustal
    DEM. Their model is a 25%-porosity sandstone, where det(F) - 1 of
    -0.07 really is 7% of pore volume closing. Here the same number is
    mostly particles rearranging in a granular stand-in for solid rock."""
    return rho_ini / (1.0 + np.clip(vol_strain, -max_compaction, max_dilatation))


def synthesize_paper(vol_strain, pos, nu=NU_DEM, E_scale=1.0):
    """Seismic properties of this DEM from its own published constants:
    Vp_ini from Table 1's E and rho, modulated by Botter Eq. (3); density
    by mass conservation; Vs from the same Poisson's ratio. Velocities in
    km/s, density in kg/m3."""
    rho_ini, E, _ = paper_layer_fields(pos)
    Vp_ini, _ = vp_vs_from_E_rho(E * E_scale, rho_ini, nu)
    Vp = vp_from_strain(vol_strain, Vp_ini / 1000.0)
    rho = density_from_mass_conservation(rho_ini, vol_strain)
    Vs = Vp / np.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))
    return rho, Vp, Vs


def paper_layer_summary(nu=NU_DEM):
    """Table 1 turned into seismic properties, layer by layer."""
    rows = []
    for z_lo, z_hi, _, _, rho, E_gpa, fric, ucs, name in ZONES_PAPER:
        Vp, Vs = vp_vs_from_E_rho(E_gpa * 1e9, rho, nu)
        rows.append(dict(name=name, z=(z_lo / 1e3, z_hi / 1e3), rho=rho, E=E_gpa,
                         friction=fric, UCS=ucs, Vp=Vp, Vs=Vs, VpVs=Vp / Vs,
                         Z=rho * Vp, mod_ratio=E_gpa * 1e3 / ucs))
    return rows
