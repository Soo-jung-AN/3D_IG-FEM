"""IG-FEM volumetric strain for one DEM stage, saved as .npy.

main.py runs the whole pipeline for one hard-coded stage and writes a
98 MB .vtk. For a sweep across the five strength models of An & So (2026)
only the strain field is wanted, so this reuses the same assembly and
solve and writes a single array.

Usage:  python3 run_stage.py m4_1 [out_dir]
"""
import sys
import time

import numpy as np
import vtk
from scipy import sparse
from scipy.sparse.linalg import splu

from Assembly3 import M_assembly_3D_block, A_assembly_3D_block, R_assembly_3D_block
from preprocessing3 import reshape_3D, Get_shf_coef_3D, Get_gp_cood_3D


def load_mesh(path="./txt/tetrahedrone.vtu"):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    cells = data.GetCells()
    cells.InitTraversal()
    tets = []
    for _ in range(data.GetNumberOfCells()):
        ids = vtk.vtkIdList()
        cells.GetNextCell(ids)
        if ids.GetNumberOfIds() == 4:
            tets.append([ids.GetId(j) for j in range(4)])
    return np.array(tets)


def vol_strain(undeformed, deformed, ele_id):
    p_num = len(undeformed)
    TT_E = len(ele_id)
    U = deformed - undeformed
    Ux, Uy, Uz = U[:, 0], U[:, 1], U[:, 2]

    SC = np.zeros((TT_E, 4, 4)); Get_shf_coef_3D(SC, ele_id, undeformed)
    PQ = np.zeros((TT_E, 4, 4)); Get_gp_cood_3D(PQ, ele_id, undeformed)

    nnz = 16 * TT_E
    M_RC = np.zeros((2, nnz), dtype=np.int64); M_data = np.zeros(nnz)
    M_assembly_3D_block(SC, ele_id, undeformed, PQ, M_RC, M_data)
    M0 = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num, p_num)).tocsc()
    del M_RC, M_data

    A_RC = np.zeros((2, nnz), dtype=np.int64)
    Axd, Ayd, Azd = (np.zeros(nnz) for _ in range(3))
    A_assembly_3D_block(SC, ele_id, undeformed, PQ, A_RC, Axd, Ayd, Azd)
    Ax = sparse.csr_matrix((Axd, (A_RC[0], A_RC[1])), shape=(p_num, p_num))
    Ay = sparse.csr_matrix((Ayd, (A_RC[0], A_RC[1])), shape=(p_num, p_num))
    Az = sparse.csr_matrix((Azd, (A_RC[0], A_RC[1])), shape=(p_num, p_num))
    del A_RC, Axd, Ayd, Azd, SC, PQ

    # R_assembly needs the shape coefficients again; they were freed above
    # to keep the two 4x4-per-element arrays from overlapping the matrices.
    SC = np.zeros((TT_E, 4, 4)); Get_shf_coef_3D(SC, ele_id, undeformed)
    PQ = np.zeros((TT_E, 4, 4)); Get_gp_cood_3D(PQ, ele_id, undeformed)
    R0 = np.zeros(p_num)
    R_assembly_3D_block(SC, ele_id, undeformed, PQ, R0)
    del SC, PQ

    lu = splu(M0)
    F11 = lu.solve(Ax @ Ux + R0)
    F22 = lu.solve(Ay @ Uy + R0)
    F33 = lu.solve(Az @ Uz + R0)
    H12 = lu.solve(Ay @ Ux); H23 = lu.solve(Az @ Uy); H31 = lu.solve(Ax @ Uz)
    H13 = lu.solve(Az @ Ux); H21 = lu.solve(Ax @ Uy); H32 = lu.solve(Ay @ Uz)

    return (F11 * (F22 * F33 - H23 * H32)
            - H12 * (H21 * F33 - H23 * H31)
            + H13 * (H21 * H32 - F22 * H31)) - 1.0


if __name__ == "__main__":
    stage = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./results"
    t0 = time.time()
    undeformed = np.loadtxt("./txt/init_pos.txt")
    deformed = np.loadtxt(f"./txt/{stage}_pos.txt")
    ele_id = reshape_3D(undeformed, load_mesh())
    print(f"{stage}: {len(undeformed)} particles, {len(ele_id)} elements "
          f"after the sliver filter", flush=True)
    vol = vol_strain(undeformed, deformed, ele_id)
    np.save(f"{out_dir}/vol_{stage}.npy", vol)
    print(f"{stage}: vol  mean {vol.mean():+.4f}  median {np.median(vol):+.4f}  "
          f"min {vol.min():+.2f}  max {vol.max():+.2f}   [{time.time()-t0:.0f} s]", flush=True)
