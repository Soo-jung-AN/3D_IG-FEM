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

**PFC 6.0 only auto-calls a file whose extension it recognises.**
`pfc3d600_console.exe model.py` loads, prints `pfc3d>` and waits
forever. It ignores extra arguments (`exe call model.py` hangs the same
way) and does not read stdin. `pfc_pipeline.write_wrapper` generates the
.dat that calls a shim that runs the model, which is the only form that
works. Three more things about that path, each of which cost a run:
`program call` on a .py leaves `__file__` UNDEFINED and `__name__` not
`"__main__"`; an error inside a called file aborts the rest of the .dat,
including its `program quit`, so an unattended batch sits at the prompt;
and PFC exits 0 even after a Python traceback, so the return code proves
nothing and only the exports do.

**PFC 6.0 embeds numpy 1.13.** No `np.trapezoid`, no `np.random.default_rng`,
no `return_indices` on `np.intersect1d`. Anything imported by a model
script has to run there as well as on a current numpy. `np.trapezoid` in
`model_spec.n_particles` broke on both, since it needs numpy 2.0.

**`emod`, `kratio`, `pb_emod` and `pb_kratio` are READ-ONLY** properties
of linearpbond — `contact property pb_kratio ...` answers `is read
only!` and aborts the run. Stiffness goes through the methods,
`contact method deformability` and `contact method pb_deformability`.
Only the strengths and `fric` are properties. Two more rejections from
the same file: `model solve ratio` is not accepted (it wants
`ratio-average`, `ratio-local` or `ratio-maximum`), and `model cycle i
calm` needs the interval, `model cycle i calm i2`. `Ball.set_group`
takes NO keyword arguments, so `set_group(name, slot="unit")` is a
TypeError; pass the slot positionally, or use the vectorised
`ballarray.set_group(mask, name, slot)`.

**Contact properties do not survive relaxation.** Cycling the pack to
push out the overlaps `ball distribute` leaves behind destroys and
recreates most of the contact list — 19,248 contacts before, 18,468
after, and the recreated ones silently revert to the cmat default.
Assign per-unit contact properties AFTER relaxing, never before. Ball
attributes do survive, and density has to be set BEFORE the first cycle:
an unassigned ball has zero inertial mass and PFC refuses to cycle.

**A contact matches `range group X slot 'unit'` if EITHER of its balls
is in X**, so an interface contact matches both units and keeps whichever
was applied LAST. On the smoke pack the per-unit applications sum to
38,553 over 32,283 contacts, so ~6,300 are interfaces decided by
application order. `build_model.assign_contacts` applies strongest
first so the weaker unit wins — otherwise dict order silently
overwrites the décollement with the underthrust section beneath it.

**A DEM wedge with no walls falls out of the domain.** `model domain
... condition destroy` deletes the escapees without a word, so the
symptom is an export of zero particles, not an error. The Nankai model
has no walls at all: y is PERIODIC (which is what makes the slab plane
strain — with `destroy` in y the pack bleeds out of the sides), the toe
and free surface are open, and the base and landward backstop are layers
of fixed particles. `model domain condition` takes one keyword per
direction: x, y, z.

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

Verified against PFC 6.00 Release 008 on this machine:

- the batch invocation, now recorded in `pfc_pipeline.json` rather than
  guessed at.
- `pfc_export.py` — all five `ballarray` accessors exist and return the
  documented shapes; the per-ball fallback names exist too.
- `nankai/build_model.py` — every `it.command(...)` string executes, and
  the sequence runs end to end to a matched pair of exports.
- `nankai/friction_sweep.py --run` — launches PFC per friction value and
  measures the result.

NOT verified, and this is now the real open question:

- the wedge is not in equilibrium. On a 6,627-ball smoke pack the
  settling solve stalls at ratio-average ~2e-2 against a 1e-5 target,
  sheds a third of its particles off the free toe, and settles to
  alpha 1.68° against the section's 2.40°. Bond strengths, the bonding
  gap and the particle size are uncalibrated. A friction sweep is not a
  calibration until the settled wedge holds the observed taper.
- `strain_analysis.py` on a pack this coarse: 69 particles end up with
  no supporting element and `splu` dies with "Factor is exactly
  singular". The mesh stage is fine. Probably a resolution artefact of
  the deliberately tiny smoke model, but it has not been shown to go
  away at full resolution.

## Conventions

- Long runs go in the background and are polled; do not block on them.
- `results/` is gitignored except the small, expensive-to-recompute
  files — see the README table. `results/vol_m4_1.npy` is the big model's
  IG-FEM strain and replaces the 94 MB .vtk everywhere.
- Report numbers, not impressions. If a claim can be checked with a
  30-second script, check it before making it.
