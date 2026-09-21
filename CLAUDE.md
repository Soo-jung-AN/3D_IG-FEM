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

**`contact method bond gap 0.0` leaves a cohesionless pack**, and the
reason is two-sided. Relaxation exists to remove the overlaps that
`gap 0.0` needs, so by the time bonding is asked almost nothing
qualifies; and a contact at a positive gap DOES NOT EXIST unless the
CMAT keeps it. From the linearpbond manual: "One can ensure the
existence of contacts between all pieces with a contact gap less than a
specified bonding gap by specifying it with the proximity in the
contact cmat default command." So `proximity` and the bonding gap are
the same number. At 0.2 r_min that bonds 63% of contacts instead of
almost none.

**A boundary of fixed particles must be several particles THICK.**
`CONVEYOR_THICKNESS` and `BACKSTOP_WIDTH` are geological lengths in
metres, which at the production r_mean of 33 m are 4.5 and 9.1 particle
diameters and fine — but at the r_mean of 200 m used for a smoke test
the conveyor was 0.75 of a diameter. A sieve, not a wall: the pack
poured through its own base and 92.5% of it was destroyed. They are now
floored at `MIN_BOUNDARY_DIAMETERS` particle diameters, which brought
that loss to 7.1% and changes nothing at production resolution.

**`it.contact.list`'s first argument is the PROCESS name**, so
`list("ball-ball")` raises `ValueError: Unknown process name`. Only that
first argument is positional — `type` and `all` must be keywords, or
PFC answers `TypeError: function takes at most 1 argument (3 given)`.
`all=True` includes virtual contacts, those inside the proximity but not
touching, which is why `it.contact.count()` reports far fewer contacts
than `contact method bond` says it applied to. `pb_state` is 0
unbonded, 1 broke in tension, 2 broke in shear, 3 bonded.

**The Nankai wedge is FRICTIONAL, not bonded** (`NANKAI_WEDGE`, default
`frictional`). A real prism stands in compression on friction, not on the
tensile strength of its own cement, and the grid in
`nankai/strength_gap_sweep.py` is the evidence: the bonding gap moves the
bonded fraction (43% at 0.05 r_min, 62% at 0.40) and barely moves alpha_0
at all, while strength moves everything — and it takes a x100 multiplier,
pb_ten of 100 MPa in the outer prism, harder than granite, before the
skeleton stops shattering. In the frictional wedge no parallel bonds are
installed at all, an unbonded linearpbond contact being identical to the
linear model, and a seeded fault is a band of LOW FRICTION
(`WEAK_FRICTION`) rather than weak cement.

**Read the bond strengths against rho*g*H, not against each other.**
The Nankai wedge is 7,900 m thick, so the stress at its base is 209 MPa,
and the strongest unit in `PROPERTIES` bonds at 10 MPa — 4.8% of it.
The rest are between 0.02% and 1.4%. Such a skeleton loses 63% -> 39% of
its bonds at ONE TENTH of gravity and ends at 16% whether gravity is
ramped over ten steps or applied in one, so a shattering pack is a
property problem and no reordering of the commands will fix it.
`build_model.strength_check()` prints this before the run starts.

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

**What alpha_0 now measures is the toe collapsing.** Stage by stage on
the frictional wedge (`nankai/fig_stages_frictional.png`, written by
`nankai/stage_figure.py` from the snapshots `NANKAI_SNAPSHOTS` drops):
built −0.063, relaxed +0.298, gravity on +1.476, settled +4.077,
converged +4.212. The landward 30–45 km sits on the digitised sea floor
at every stage; the whole error is the trenchward 20 km, which sinks 1–2
km. The toe is thin, so its confining stress and therefore its frictional
resistance are small, while the landward component of the tilted gravity
does not care how thick the wedge is. Two things the model does not have
would hold it: sea-water buoyancy, and the convergence that in reality
never stops. The settling stage switches convergence OFF and then asks
the wedge to keep its shape, which is not a state a real prism is ever
in.

NOT verified, and this is now the real open question:

- the frictional wedge sheds its toe: alpha_0 ends +4.08° off the
  section with 9.8% of particles lost, and the solve stops at its cycle
  cap rather than reaching ratio-average 1e-4, so it is creeping. The
  likely fixes are physical, not procedural — drive the conveyor during
  settling instead of after it, and give the model the sea water it is
  sitting in.
- the density convention is unresolved. `PROPERTIES` calls its densities
  bulk values, but PFC takes them as PARTICLE densities, so at the
  measured porosity of 0.397 the pack weighs 60% of what the table says
  (outer_prism 2100 -> 1267). That happens to land near the buoyant
  weight of saturated sediment (2100 - 1025 = 1075), so the model may be
  accidentally approximating a buoyancy it does not model. Downstream,
  `vp_from_strain` reads the exported value as a bulk density, where the
  same number is 66% too high.
- `friction_sweep.report()` now refuses to name a calibrated mu_b until
  alpha_0 agrees across the swept values to `ALPHA0_TOLERANCE` (0.05°).
  On the last sweep it spread 1.672°. Do not quote a crossing until
  that gate passes.

## Conventions

- Long runs go in the background and are polled; do not block on them.
- `results/` is gitignored except the small, expensive-to-recompute
  files — see the README table. `results/vol_m4_1.npy` is the big model's
  IG-FEM strain and replaces the 94 MB .vtk everywhere.
- Report numbers, not impressions. If a claim can be checked with a
  30-second script, check it before making it.
