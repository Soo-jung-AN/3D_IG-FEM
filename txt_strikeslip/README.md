# Strike-slip DEM run

A second, smaller model than the extension models in `txt/`: 52,995
particles in a 1.97 x 1.97 x 0.97 km box, radii 20-30 m, median
centre-to-centre spacing 41.2 m.

| file | contents |
|---|---|
| `init_pos3.txt` | undeformed particle centres, (x, y, z) in m |
| `pos_3.txt` | deformed centres, same ordering |
| `rad_3.txt` | particle radii, m |
| `density_3.txt` | per-particle bulk density, kg/m3 |
| `contactForce_3.txt` | per-particle contact force, 3 components, N |
| `StrikeSlip_vtk3575.vtk` | the ParaView INPUT: the same undeformed centres as a point cloud (CELL_TYPES = 1, VTK_VERTEX), carrying `radius` and `velocity_vectors`. It holds no tetrahedra, so it is not a mesh — it is what you feed to ParaView's Delaunay3D to make one. |

## Layers

Four density layers, top down. The 0 to -100 m layer is a deliberate
stiff cap, which is why it is denser than the layer beneath it:

| depth (m) | rho (kg/m3) | particles |
|---|---|---|
| -14 to -100 | 2625 | 5,219 |
| -100 to -300 | 2300 | 10,611 |
| -300 to -600 | 2500 | 15,944 |
| -600 to -986 | 2700 | 21,221 |

## It is strike-slip, not extension

The bounding box widens in x from 1971 to 2200 m, but that is a sheared
square becoming a parallelogram, not the material stretching: u_x
averages -36.3 m in the lower half of y and +36.5 m in the upper half, a
shear couple whose mean is near zero, and the interior E11 is only
-0.009. Do not read the box widening as a stretching factor.

A fault forms at y = 0.9-1.2 km with a 225 m core (FWHM) whose trace
wanders 475 m across strike. Bulk shear strain is gamma = 0.12.

## Known problems with the run

The cap layer is unconfined and bulking: in the top 100 m the volumetric
strain is +0.125 and the vertical strain E33 is +0.171, against +0.020
and +0.039 below it, and its particles still move at 5.3x the interior
median speed. Pass `--zmax -100` to cut it out of the mesh. The snapshot
is also not fully quasi-static (mean v_z is downward, max/median |v| is
19.4).

## Reproducing the analysis

```
python3 vp_from_strain.py --init txt_strikeslip/init_pos3.txt \
    --pos txt_strikeslip/pos_3.txt --density txt_strikeslip/density_3.txt \
    --alpha 125 --zmax -100 --out results/vp_model3_nocap.npz
python3 strain_analysis.py --init txt_strikeslip/init_pos3.txt \
    --pos txt_strikeslip/pos_3.txt --alpha 125 --out results/strain_model3.npz
python3 seismic_section.py results/vp_model3_nocap.npz model3_nocap
python3 coherence_test.py
```
