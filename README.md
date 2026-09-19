# 3D_IG-FEM
3-dimensional IG-FEM

<img width="1353" height="1250" alt="Image" src="https://github.com/user-attachments/assets/71644e9b-10cc-46df-9956-970df1b96dbe" />




3D Delanauy triangulation is required to generate vtu mesh domain (e.g., tetrahedrone.vtu in the "main.py") using an open source visualization software, Paraview (https://www.paraview.org/).

The 9*p_num sparse system is block diagonal: all nine of its mass-matrix
blocks are the same p_num x p_num matrix, and the gradient operator has only
three distinct blocks (d/dx, d/dy, d/dz). main.py assembles one copy of each
and reuses a single LU factorisation for the nine right-hand sides, so the
solve is p_num-sized rather than 9*p_num-sized. That is small enough for
scipy's sparse solver; PyPardiso is no longer required.

Please install "vtk" following code in terminal before running the "main.py"

"pip install vtk"

(numpy, scipy and matplotlib are also required.)

# Memory & Runtime overview

Total discrete element particles         : 259,943

Total tetrahedral mesh elements          : 1,550,208 (1,469,046 after the
                                           sliver filter, see below)

RAM usage (peak)                         : 8.5 GB, whole pipeline including
                                           the .vtk export

Elapsed time (IG-FEM solving)            : 967 sec

Elapsed time (VTKUnstructuredConverter2) : 6 sec

The assembly loops are plain Python (the numba decorators in Assembly3.py /
preprocessing3.py are commented out), so the ~16 min is dominated by the
element loop, not by the linear algebra. Re-enabling numba would cut it
substantially.

Earlier versions of this code needed ~17 GB to solve and ~35 GB to write the
.vtk, and were run on a 32-core Xeon workstation with 128 GB RAM. Both
figures were artefacts rather than requirements: the solve was assembling
nine redundant copies of the same system, and the writer buffered the entire
output file as a list of Python strings before writing it. The numbers above
were measured on an ordinary 4-core, 15 GB container.

# Sliver filtering

reshape_3D drops degenerate tetrahedra by the scale-invariant shape quality
q = 6*sqrt(2)*V / L_max^3 (1 for a regular tetrahedron, 0 for a sliver),
with a default threshold of q_min = 0.05. The previous test was on absolute
volume (< 1e-6), which is scale dependent and removed 0 of the 1,550,208
elements of this mesh even though ~5% of them are slivers. Those slivers are
where the recovered deformation gradient blows up: without the filter, vol
ranged from -392 to +211 against a median of -0.07. See the docstring in
preprocessing3.py for the threshold sweep.

# What is in results/

Most of `results/` is ignored, because three of the grid-based `.npz`
files are 192-293 MB and GitHub rejects anything over 100 MB. What is
committed is the set that is small and expensive to recompute:

| file | size | what it is | cost to regenerate |
|---|---|---|---|
| `vol_m1_1.npy` … `vol_m5_1.npy` | 2 MB each | IG-FEM volumetric strain, An & So (2026) models M1-M5 at the first extension stage, 259,943 particles, sliver filter q >= 0.05 | 22 min each |
| `strain_model3.npz` | 7.2 MB | full strain tensor of the strike-slip run: F, E, principal strains, max shear, polar-decomposition rotation | 2 min |
| `vp_model3.npz` | 4.5 MB | Vp/Vs/rho of the strike-slip run, all 52,995 particles | 2 min |
| `vp_model3_nocap.npz` | 4.1 MB | the same with the bulking cap layer cut from the mesh (47,802 particles) | 2 min |
| `trace_nocap.npy`, `fault_trace_model3.npy` | < 0.1 MB | the picked fault trace and its half-width per x-column | seconds |

`vol_m4_1.npy` is the one that matters most: it is byte-for-byte the
`vol` field of the 93.6 MB `results/80-3.vtk` that `main.py` writes
(verified to 1.9e-15), so every downstream script can take it directly
and skip both the 16-minute solve and the large .vtk:

```
python3 compare_3d.py --igfem-vol results/vol_m4_1.npy
```

## Regenerating the files that are not committed

All of them are fast once `vol_m4_1.npy` is in place:

```
python3 compare_3d.py --igfem-vol results/vol_m4_1.npy   # compare_3d.npz,  293 MB, 116 s
python3 plausibility_3d.py                                # plausibility_3d.npz, 192 MB
python3 paper_calibration.py                              # paper_calibration.npz, 280 MB
python3 main.py                                           # results/80-3.vtk, 94 MB, 967 s
```

`main.py` is only needed for the .vtk itself; nothing else in the
repository reads it.

The strike-slip inputs (`init_pos3.txt`, `pos_3.txt`, `density_3.txt`,
`rad_3.txt`, `contactForce_3.txt`) are not in the repository either.
Point the scripts at wherever you keep them:

```
python3 vp_from_strain.py --init <init_pos3.txt> --pos <pos_3.txt> \
    --density <density_3.txt> --alpha 125 --zmax -100 \
    --out results/vp_model3_nocap.npz
```
