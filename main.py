import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import numpy.linalg as lina
from Assembly3 import M_assembly_3D_block, A_assembly_3D_block, R_assembly_3D_block
from preprocessing3 import reshape_3D, Get_shf_coef_3D, Get_gp_cood_3D, remove_unused_nodes
from rock_physics import synthesize_vpvs
from scipy import sparse
from scipy.sparse.linalg import splu
import vtk
np.set_printoptions(precision=10, threshold=20000000, linewidth=20000000)
############################################################################################################################################################
def VTKUnstructuredConverter2(points, rad, E11, E22, E33, E12, E13, E23, vol, distot, stresses, DynStress, Z_disp, components, phi_synth, Vp_synth, Vs_synth, VpVs_synth):
    num_points = points.shape[0]

    def write_scalar(f, name, arr):
        f.write(f'SCALARS {name} float 1\n')
        f.write('LOOKUP_TABLE default\n')
        f.write('\n'.join(map(str, arr.ravel())))
        f.write('\n')

    # Streamed straight to disk. Buffering the whole file as a list of Python
    # strings first needs tens of GB at this particle count; writing block by
    # block only ever holds one field in memory.
    with open("./results/80-3.vtk", 'w') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write('Unstructured Grid Example\n')
        f.write('ASCII\n')
        f.write('DATASET UNSTRUCTURED_GRID\n')

        f.write(f'POINTS {num_points} float\n')
        for x, y, z in points:
            f.write(f'{x} {y} {z}\n')

        f.write(f'CELLS {num_points} {num_points * 2}\n')
        for i in range(num_points):
            f.write(f'1 {i}\n')

        f.write(f'CELL_TYPES {num_points}\n')
        f.writelines('1\n' for _ in range(num_points))

        f.write(f'POINT_DATA {num_points}\n')
        for name, arr in (('rad', rad), ('E11', E11), ('E22', E22), ('E33', E33),
                          ('E12', E12), ('E13', E13), ('E23', E23),
                          ('vol', vol), ('distot', distot), ('stress', stresses),
                          ('DynStress', DynStress), ('Z_disp', Z_disp),
                          ('components', components), ('phi_synth', phi_synth),
                          ('Vp_synth', Vp_synth), ('Vs_synth', Vs_synth),
                          ('VpVs_synth', VpVs_synth)):
            write_scalar(f, name, arr)
    print("vtkGenerationisDone")

undeformed_cood = np.loadtxt("./txt/init_pos.txt") 
deformed_cood = np.loadtxt("./txt/m4_1_pos.txt") 
ContactForce = np.loadtxt('./txt/m4_1_contactF.txt')
rad = np.loadtxt("./txt/init_rad.txt") 
p_num = len(undeformed_cood); print('the number of particles',p_num)
###########################
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("./txt/tetrahedrone.vtu")
reader.Update()
data = reader.GetOutput()
num_cells = data.GetNumberOfCells()
print(f"Total Elements: {num_cells}")
points = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
cells = data.GetCells()
cells.InitTraversal()
tetra_indices = []
for _ in range(num_cells):
    id_list = vtk.vtkIdList()
    cells.GetNextCell(id_list)
    if id_list.GetNumberOfIds() == 4:
        tetra_indices.append([id_list.GetId(j) for j in range(4)])
tetra_indices = np.array(tetra_indices)
TT_E = len(tetra_indices)
print(f"Tetrahedral Elements Shape: {tetra_indices.shape}")
ele_id = tetra_indices
print(len(ele_id))
ele_id = reshape_3D(undeformed_cood, ele_id)
print('after',len(ele_id))
U = deformed_cood - undeformed_cood
Ux, Uy, Uz = U[:,0], U[:,1], U[:,2]
Z_disp = Uz
#################################################################################
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")
# for tet in tetra_indices:
#     tetra_points = points[tet]
#     verts = [[tetra_points[j] for j in [0, 1, 2]], 
#              [tetra_points[j] for j in [0, 1, 3]], 
#              [tetra_points[j] for j in [1, 2, 3]], 
#              [tetra_points[j] for j in [0, 2, 3]]]s
#     ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, edgecolor="k"))
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", marker="o", s=10)
# ax.set_xlabel("X-axis");ax.set_ylabel("Y-axis");ax.set_zlabel("Z-axis");ax.set_title("3D Tetrahedral Mesh from VTU")
# plt.show()
#################################################################################
# Implicit-Global finite element method
import time
solving_time = time.time()
SC_mat_e = np.zeros((TT_E, 4, 4), dtype=np.float64)
Get_shf_coef_3D(SC_mat_e, ele_id, undeformed_cood)
PQ_detJ_e = np.zeros((TT_E, 4, 4), dtype=np.float64)
Get_gp_cood_3D(PQ_detJ_e, ele_id, undeformed_cood)

# The full 9*p_num system is block diagonal with only a handful of distinct
# blocks, so assemble one copy of each and reuse a single factorisation for
# all nine components (see the note in Assembly3.py). Identical result, ~6x
# less assembly memory, and a p_num-sized solve instead of a 9*p_num one.
nnz = 16 * TT_E
M_RC = np.zeros((2, nnz), dtype=np.int64); M_data = np.zeros(nnz, dtype=np.float64)
M_assembly_3D_block(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, M_RC, M_data)
M0 = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num, p_num)).tocsc()
del M_RC, M_data

A_RC = np.zeros((2, nnz), dtype=np.int64)
Ax_data = np.zeros(nnz, dtype=np.float64)
Ay_data = np.zeros(nnz, dtype=np.float64)
Az_data = np.zeros(nnz, dtype=np.float64)
A_assembly_3D_block(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, A_RC, Ax_data, Ay_data, Az_data)
Ax = sparse.csr_matrix((Ax_data, (A_RC[0], A_RC[1])), shape=(p_num, p_num))
Ay = sparse.csr_matrix((Ay_data, (A_RC[0], A_RC[1])), shape=(p_num, p_num))
Az = sparse.csr_matrix((Az_data, (A_RC[0], A_RC[1])), shape=(p_num, p_num))
del A_RC, Ax_data, Ay_data, Az_data

R0 = np.zeros(p_num, dtype=np.float64)
R_assembly_3D_block(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, R0)

# One factorisation of the mass matrix, reused for all nine right-hand sides.
# M0^-1 R0 == 1 exactly (partition of unity), so adding R0 to the diagonal
# components turns them into the deformation gradient F_ii = 1 + H_ii, while
# the off-diagonal components stay as the displacement gradient H_ij.
lu = splu(M0)
F11 = lu.solve(Ax @ Ux + R0)
F22 = lu.solve(Ay @ Uy + R0)
F33 = lu.solve(Az @ Uz + R0)
H12 = lu.solve(Ay @ Ux)
H23 = lu.solve(Az @ Uy)
H31 = lu.solve(Ax @ Uz)
H13 = lu.solve(Az @ Ux)
H21 = lu.solve(Ax @ Uy)
H32 = lu.solve(Ay @ Uz)

H11, H22, H33 = F11 - 1, F22 - 1, F33 - 1

# Green-Lagrangian strain E = 0.5*(H + H^T + H^T H), in H-components:
#   E_ij = 0.5*(H_ij + H_ji) + 0.5 * sum_k H_ki H_kj
E11 = H11 + 0.5 * (H11**2 + H21**2 + H31**2)
E22 = H22 + 0.5 * (H12**2 + H22**2 + H32**2)
E33 = H33 + 0.5 * (H13**2 + H23**2 + H33**2)
E12 = 0.5 * (H12 + H21) + 0.5 * (H11*H12 + H21*H22 + H31*H32)
E13 = 0.5 * (H13 + H31) + 0.5 * (H11*H13 + H21*H23 + H31*H33)
E23 = 0.5 * (H23 + H32) + 0.5 * (H12*H13 + H22*H23 + H32*H33)
E21 = E12
E31 = E13
E32 = E23

# Volumetric strain (dilatation) = det(F) - 1, with F = I + H. F11/F22/F33
# already carry the +1 from R0; the off-diagonals of I are zero so H_ij = F_ij.
vol = (F11 * (F22 * F33 - H23 * H32)
       - H12 * (H21 * F33 - H23 * H31)
       + H13 * (H21 * H32 - F22 * H31)) - 1
tr = (E11 + E22 + E33) / 3
distot = 0.5 * (((E11 - tr) * (E22 - tr) * (E33 - tr)) - E21**2 - E32**2 - E31**2)

Zpos = undeformed_cood[:,2]; Xpos = undeformed_cood[:,0]
ContactForce_means = np.mean(ContactForce, axis=1)
stresses = np.zeros((p_num))
rhogh = np.zeros((p_num))
contactForceWithDepths = np.zeros((p_num))

for i in range(p_num):
    z = Zpos[i]; x = Xpos[i]
    if -15e3 <= z < -11e3:
        rhogh[i] = 2700 * 10 * abs(z)
        contactForceWithDepths[i] = ContactForce_means[i] / (np.pi * rad[i] **2)
    if -11e3 <= z < -7e3:
        rhogh[i] = 2500 * 10 * abs(z)
        contactForceWithDepths[i] = ContactForce_means[i] / (np.pi * rad[i] **2)
    if -7e3 <= z < 0:
        rhogh[i] = 2300 * 10 * abs(z)
        contactForceWithDepths[i] = ContactForce_means[i] / (np.pi * rad[i] **2)
    if -15e3 <= z < -13e3 and 30e3 <= x <= 120e3:
        rhogh[i] = 2100 * 10 * abs(z)
        contactForceWithDepths[i] = ContactForce_means[i] / (np.pi * rad[i] **2)
##########################
Zpos = undeformed_cood[:,2]; Xpos = undeformed_cood[:,0]
components = np.zeros((p_num))
for i in range(p_num):
    zz = Zpos[i]
    z0 = 0
    z1 = -1e3
    z2 = -3e3
    z3 = -5e3
    z4 = -7e3
    z5 = -9e3 
    z6 = -11e3
    z7 = -13e3
    z8 = -15e3
    if z0 >= zz > z1 : components[i]=1
    elif z1 >= zz > z2 : components[i]=2
    elif z2 >= zz > z3 : components[i]=3
    elif z3 >= zz > z4 : components[i]=4
    elif z4 >= zz > z5 : components[i]=5
    elif z5 >= zz > z6 : components[i]=6
    elif z6 >= zz > z7 : components[i]=7
    elif z7 >= zz > z8 : components[i]=8

for i in range(p_num):
    xx = Xpos[i]
    zz = Zpos[i]
    if 30e3 < xx < 120e3 and zz<-13e3:
       components[i]=9

##########################
# Synthetic Vp/Vs from DEM finite volumetric strain, following
# Botter et al. (2014, Marine and Petroleum Geology 57, 187-207), Eqs. 1-4.
# `vol` (= det(F) - 1) computed above is exactly the volumetric strain
# (dilatation) used by that workflow. phi_ini / Vp_ini below are example
# reference (undeformed) properties per depth zone -- reusing the same
# grain densities already assumed for the DynStress calculation above --
# and should be calibrated to the actual DEM materials, the same way
# Botter et al. calibrated their sandstone/shale properties (their Table 3).
phi_ini_arr = np.zeros(p_num)
rho_g_arr = np.zeros(p_num)
Vp_ini_arr = np.zeros(p_num)   # km/s

zone1 = (Zpos >= -15e3) & (Zpos < -11e3)
zone2 = (Zpos >= -11e3) & (Zpos < -7e3)
zone3 = (Zpos >= -7e3) & (Zpos < 0)
zone4 = (Zpos >= -15e3) & (Zpos < -13e3) & (Xpos >= 30e3) & (Xpos <= 120e3)

rho_g_arr[zone1], phi_ini_arr[zone1], Vp_ini_arr[zone1] = 2700.0, 0.10, 4.0
rho_g_arr[zone2], phi_ini_arr[zone2], Vp_ini_arr[zone2] = 2500.0, 0.15, 3.0
rho_g_arr[zone3], phi_ini_arr[zone3], Vp_ini_arr[zone3] = 2300.0, 0.25, 2.0
rho_g_arr[zone4], phi_ini_arr[zone4], Vp_ini_arr[zone4] = 2100.0, 0.35, 1.5

phi_synth, rho_synth, Vp_synth, Vs_synth, VpVs_synth = synthesize_vpvs(
    vol, phi_ini_arr, rho_g_arr, Vp_ini_arr
)
print("---Synthetic Vp/Vs (Botter et al., 2014) ---")
print("phi   : min %.4f  mean %.4f  max %.4f" % (phi_synth.min(), phi_synth.mean(), phi_synth.max()))
print("Vp    : min %.4f  mean %.4f  max %.4f (km/s)" % (Vp_synth.min(), Vp_synth.mean(), Vp_synth.max()))
print("Vs    : min %.4f  mean %.4f  max %.4f (km/s)" % (Vs_synth.min(), Vs_synth.mean(), Vs_synth.max()))
print("Vp/Vs : min %.4f  mean %.4f  max %.4f" % (VpVs_synth.min(), VpVs_synth.mean(), VpVs_synth.max()))

DynStress = contactForceWithDepths - rhogh
print(np.mean(rhogh))

plt.plot(DynStress) 
plt.show()

print("---IG-FEM strain calculation is done within",time.time()-solving_time,"sec")

start = time.time()
VTKUnstructuredConverter2(deformed_cood, rad, E11, E22, E33, E12, E13, E23, vol, distot, stresses, DynStress, Z_disp, components, phi_synth, Vp_synth, Vs_synth, VpVs_synth)
print("---VTK convert is done within",time.time()-solving_time,"sec")

print("---3D IG-FEM strain calculation complete---")
print("Computed F11 Tensor (First 3 values):", F11[:3])
print("Computed E11 Tensor (First 3 values):", E11[:3])
print("Computed E12 Tensor (First 3 values):", E12[:3])
print("Computed E33 Tensor (First 3 values):", E33[:3])
print("Computed E22 Tensor (First 3 values):", E22[:3])
print("Computed vol Tensor (First 3 values):", vol[:3])
print("Computed distot Tensor (First 3 values):", distot[:3])