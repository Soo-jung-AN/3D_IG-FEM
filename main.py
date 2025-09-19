import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
import numpy.linalg as lina
from Assembly3 import M_assembly_3D, A_assembly_3D, R_assembly_3D
from preprocessing3 import reshape_3D, Get_shf_coef_3D, Get_gp_cood_3D, remove_unused_nodes
from scipy import sparse
from pypardiso import spsolve
import vtk
np.set_printoptions(precision=10, threshold=20000000, linewidth=20000000)
############################################################################################################################################################
def VTKUnstructuredConverter2(points, rad, E11, E22, E33, E12, E13, E23, vol, distot, stresses, DynStress, Z_disp, components):
    num_points = points.shape[0]
    lines = []
    lines.append('# vtk DataFile Version 2.0\n')
    lines.append('Unstructured Grid Example\n')
    lines.append('ASCII\n')
    lines.append('DATASET UNSTRUCTURED_GRID\n')

    lines.append(f'POINTS {num_points} float\n')
    for x, y, z in points:
        lines.append(f'{x} {y} {z}\n')

    lines.append(f'CELLS {num_points} {num_points * 2}\n')
    for i in range(num_points):
        lines.append(f'1 {i}\n')

    lines.append(f'CELL_TYPES {num_points}\n')
    lines.extend(['1\n'] * num_points)

    lines.append(f'POINT_DATA {num_points}\n')
    lines.append('SCALARS rad float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    rad_flat = rad.ravel()
    lines.append('\n'.join(map(str, rad_flat)) + '\n')

    lines.append('SCALARS E11 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E11_flat = E11.ravel()
    lines.append('\n'.join(map(str, E11_flat)) + '\n')

    lines.append('SCALARS E11 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E11_flat = E11.ravel()
    lines.append('\n'.join(map(str, E11_flat)) + '\n')

    lines.append('SCALARS E22 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E22_flat = E22.ravel()
    lines.append('\n'.join(map(str, E22_flat)) + '\n')

    lines.append('SCALARS E33 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E33_flat = E33.ravel()
    lines.append('\n'.join(map(str, E33_flat)) + '\n')

    lines.append('SCALARS E12 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E12_flat = E12.ravel()
    lines.append('\n'.join(map(str, E12_flat)) + '\n')

    lines.append('SCALARS E13 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E13_flat = E13.ravel()
    lines.append('\n'.join(map(str, E13_flat)) + '\n')

    lines.append('SCALARS E23 float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    E23_flat = E23.ravel()
    lines.append('\n'.join(map(str, E23_flat)) + '\n')

    lines.append('SCALARS vol float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    vol_flat = vol.ravel()
    lines.append('\n'.join(map(str, vol_flat)) + '\n')

    lines.append('SCALARS distot float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    distot_flat = distot.ravel()
    lines.append('\n'.join(map(str, distot_flat)) + '\n')
    
    lines.append('SCALARS stress float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    stress_flat = stresses.ravel()
    lines.append('\n'.join(map(str, stress_flat)) + '\n')
    
    lines.append('SCALARS DynStress float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    DynStress_flat = DynStress.ravel()
    lines.append('\n'.join(map(str, DynStress_flat)) + '\n')

    lines.append('SCALARS Z_disp float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    Z_disp_flat = Z_disp.ravel()
    lines.append('\n'.join(map(str, Z_disp_flat)) + '\n')

    lines.append('SCALARS components float 1\n')
    lines.append('LOOKUP_TABLE default\n')
    components_flat = components.ravel()
    lines.append('\n'.join(map(str, components_flat)) + '\n')

    
    with open("./results/80-3.vtk", 'w') as outFile:
        outFile.writelines(lines)
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

M_RC = np.zeros((2, 200 * TT_E), dtype=np.int64); M_data = np.zeros(200 * TT_E, dtype=np.float64)
M_assembly_3D(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, M_RC, M_data, p_num)
M_CSR = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num * 9, p_num * 9))

A_RC = np.zeros((2, 200 * TT_E), dtype=np.int64); A_data = np.zeros(200 * TT_E, dtype=np.float64)
A_assembly_3D(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, A_RC, A_data, p_num)
A_CSR = sparse.csr_matrix((A_data, (A_RC[0], A_RC[1])), shape=(p_num * 9, p_num * 9))

U = np.hstack((Ux, Uy, Uz))
U = np.hstack((U,U,U))
#U = np.column_stack((Ux, Uy, Uz, Ux, Uy, Uz, Ux, Uy, Uz)).flatten()
AU = A_CSR * U 

R_vec = np.zeros(p_num * 9, dtype=np.float64)
R_assembly_3D(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, R_vec, p_num)

AU_R = AU + R_vec

solved_F = spsolve(M_CSR, AU_R)
#from scipy.sparse.linalg import cg  # Conjugate Gradient
#solved_F, info = cg(M_CSR, AU_R, tol=1e-8, maxiter=1000)

F11 = solved_F[:p_num]
F12 = solved_F[p_num:2*p_num]
F13 = solved_F[2*p_num:3*p_num]
F21 = solved_F[3*p_num:4*p_num]
F22 = solved_F[4*p_num:5*p_num]
F23 = solved_F[5*p_num:6*p_num]
F31 = solved_F[6*p_num:7*p_num]
F32 = solved_F[7*p_num:8*p_num]
F33 = solved_F[8*p_num:]

E11 = F11 + 0.5 * (F11 **2 + F21 **2 + F31**2)
E22 = F22 + 0.5 * (F12 **2 + F22 **2 + F32**2)
E33 = F33 + 0.5 * (F13 **2 + F23 **2 + F33**2)
E12 = 0.5 * (F12 + F21) + 0.5 * (F11*F12 + F21*F22 + F31*F32)
E13 = 0.5 * (F13 + F31) + 0.5 * (F11*F13 + F21*F13 + F31*F33)
E23 = 0.5 * (F23 + F32) + 0.5 * (F12*F13 + F22*F23 + F32*F33)
E21 = E12
E31 = E13
E32 = E23

# Green Lagrangian Strain tensor , Volumetric tensor
vol = (F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)) - 1
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


DynStress = contactForceWithDepths - rhogh
print(np.mean(rhogh))

plt.plot(DynStress) 
plt.show()

print("---IG-FEM strain calculation is done within",time.time()-solving_time,"sec")

start = time.time()
VTKUnstructuredConverter2(deformed_cood, rad, E11, E22, E33, E12, E13, E23, vol, distot, stresses, DynStress, Z_disp, components)
print("---VTK convert is done within",time.time()-solving_time,"sec")

print("---3D IG-FEM strain calculation complete---")
print("Computed F11 Tensor (First 3 values):", F11[:3])
print("Computed E11 Tensor (First 3 values):", E11[:3])
print("Computed E12 Tensor (First 3 values):", E12[:3])
print("Computed E33 Tensor (First 3 values):", E33[:3])
print("Computed E22 Tensor (First 3 values):", E22[:3])
print("Computed vol Tensor (First 3 values):", vol[:3])
print("Computed distot Tensor (First 3 values):", distot[:3])