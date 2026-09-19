# 3D_IG-FEM

DEM particle displacements → finite strain → seismic velocity →
synthetic seismic section. Two model families live here: the crustal
extension models of An & So (2026) in `txt/`, and a strike-slip run in
`txt_strikeslip/`. `nankai/` is a subduction wedge being built now.

Work happens on `claude/dem-strain-vpvs-synthesis-jqdszc`, not `main`.

## The chain

```
DEM               pfc_export.py (inside PFC) → five .txt files
mesh              make_mesh.py               → .vtu   (or built in process)
strain            strain_analysis.py         → full tensor
                  vp_from_strain.py          → Vp, Vs, rho
seismic           seismic_section.py         → velocity + synthetic sections
                  coherence_test.py          → 3D semblance slices
all of it         pfc_pipeline.py            → one command, skippable stages
```

## Traps already paid for

Each of these cost real debugging time. Do not rediscover them.

**scipy's 3D Delaunay returns arbitrary orientation** — 50% negative
volume on a DEM pack. `Assembly3.py` weights element integrals by det(J)
with no absolute value, so negative-volume elements SUBTRACT mass, the
mass matrix goes indefinite, and det(F) − 1 comes back around 1e10.
`vp_from_strain.orient_positive()` fixes it. scipy's 2D Delaunay is
counter-clockwise by construction and the ParaView mesh in `txt/` is
consistently oriented, so only the raw-3D-Delaunay path is exposed.

**Do not use vtkDelaunay3D** (= ParaView's Delaunay3D filter) on a DEM
pack: it drops 14.5% of the points to "degenerate triangles" regardless
of Tolerance or Offset, and those particles then have no supporting
element. Qhull loses none and matches the ParaView mesh to 0.9999999986
on the recovered strain. See the `make_mesh.py` docstring.

**A Delaunay triangulation fills the convex hull**, so a pack with a free
surface gets long thin elements bridging the empty space. Filter on a
maximum edge length (`--alpha`) as well as on sliver shape quality.

**Filter slivers on shape, not volume.** `q = 6√2·V / L³max` is scale
invariant; an absolute volume test removed 0 of 1,550,208 elements on a
model whose element volumes run 1e4–1e8. q ≥ 0.05 is the calibrated
default (see the `preprocessing3.reshape_3D` docstring).

**Botter et al. Eq. (2) is inert at crustal porosity.** It routes density
through a porosity change, which is right for their 25%-porosity
sandstone and does nothing at 1–3%. Use
`rock_physics.density_from_mass_conservation`, ρ = ρ₀/det(F), bounded
ASYMMETRICALLY: rock can only densify by closing the porosity it has,
but can dilate much further by fracturing.

**Han (1986) in Eq. (4) breaks below 3 km/s.** It returns Vs = 0 at
Vp = 0.991 km/s. But raising the velocity is what fixes a bad Poisson's
ratio, not swapping the relation — Brocher (2005) is *softer* still at
low Vp. Both exceed Vp/Vs = 2.45 below ~2.5 km/s.

**Hertz–Mindlin is the wrong effective medium for a bonded pack.** It has
no cementation term, so it lands 34–39% below the elastic velocities of
a model whose bonds carry most of the modulus. It also pins Vp/Vs to a
single value by construction.

**Tie the wavelet frequency to the particle spacing, not the grid.**
f_max = Vp / (4 × spacing). A fine interpolation grid adds no
information the pack does not have. The diagnostic: at 30 Hz, two
velocity models differing 60% in velocity correlated at +0.993.

**Apply a datum before measuring traveltime on a synthetic section.**
Measuring from the ragged top of each trace turned a 4 ms velocity
pull-down into a 21 ms one — the rest was topographic static.

**A raw correlation between two synthetic sections is dominated by the
reference velocity model they share** — 97–99% of it on the 3D crustal
model. Build the zero-strain control and regress it out
(`section_attribution.py`) before comparing strain methods through an
image.

**Chord ≠ regression.** `nankai/geometry.summary()` reports the surface
slope as a two-point chord; the runs are measured by least squares, and
the sea floor is convex, so they differ by 0.3°. Compare against
`nankai.taper.target_alpha()`, which measures the section the same way a
run is measured.

## What is verified and what is not

Verified here, with numbers in the commit messages: the mesh against the
ParaView mesh; the strain against an independent SSPX estimate; the
pipeline stages 2–5 reproducing `results/vp_model3_nocap.npz` exactly;
`nankai/taper.py` recovering imposed surface tilts to 0.000°.

NOT verified, because there is no PFC or ParaView in the environment
this was written in:

- `pfc_export.py` — the `itasca` accessors. It tries `ballarray` first
  and a per-ball loop second, and raises naming what it tried.
- `nankai/build_model.py` — every `it.command(...)` string. PFC 6/7
  syntax, never executed.
- the PFC batch invocation itself, which differs across PFC 6/7/8 and
  comes from `pfc_pipeline.json`, `--pfc-exe` or `PFC_EXE`.

Fix these locally and the rest holds.

## Conventions

- Long runs go in the background and are polled; do not block on them.
- `results/` is gitignored except the small, expensive-to-recompute
  files — see the README table. `results/vol_m4_1.npy` is the big model's
  IG-FEM strain and replaces the 94 MB .vtk everywhere.
- Report numbers, not impressions. If a claim can be checked with a
  30-second script, check it before making it.
