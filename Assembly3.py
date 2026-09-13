
import numpy.linalg as lina
import numpy as np
#from numba import njit, float64, int64, jit
# SC_mat_e : shape func. coeff. of each element

#@njit("void(float64[:,:,::1],int64[:,::1],float64[:,::1],float64[:,:,::1],int64[:,::1],float64[::1],int64)")
#@njit("void(float64[:,:,::1], int64[:,::1], float64[::1,::1], float64[:,:,::1], int64[::1,:], float64[:], int64)")
def M_assembly_3D(SC_mat_e, ele_id, init_pos, PQ_detJ_e, M_RC, M_data, p_num): 
    count_sparse = 0
    TT_E = len(ele_id)

    for ele in range(TT_E):
        nodes = ele_id[ele]
        P_e = PQ_detJ_e[ele,0]
        Q_e = PQ_detJ_e[ele,1]
        R_e = PQ_detJ_e[ele,2]
        J_e = PQ_detJ_e[ele,3]

        SC_mat = SC_mat_e[ele]
        for i in range(4):  
            row = nodes[i]
            c1, c2, c3, c4 = SC_mat[:, i]
            for j in range(4):
                col = nodes[j]
                NN = 0
                for k in range(4): 
                    P = P_e[k]
                    Q = Q_e[k]
                    R = R_e[k]
                    J = J_e[k]
                    Ni_with_gp = c1 + c2*P + c3*Q + c4*R
                    Nj_with_gp = SC_mat[0, j] + SC_mat[1, j]*P + SC_mat[2, j]*Q + SC_mat[3, j]*R
                    NN += 1/4 * Ni_with_gp * Nj_with_gp * J

                M_RC[0, count_sparse] = row
                M_RC[1, count_sparse] = col
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + p_num
                M_RC[1, count_sparse] = col + p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 2 * p_num
                M_RC[1, count_sparse] = col + 2 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 3 * p_num
                M_RC[1, count_sparse] = col + 3 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 4 * p_num
                M_RC[1, count_sparse] = col + 4 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 5 * p_num
                M_RC[1, count_sparse] = col + 5 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 6 * p_num
                M_RC[1, count_sparse] = col + 6 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 7 * p_num
                M_RC[1, count_sparse] = col + 7 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                M_RC[0, count_sparse] = row + 8 * p_num
                M_RC[1, count_sparse] = col + 8 * p_num
                M_data[count_sparse] = NN
                count_sparse += 1
                
#@njit("void(float64[:,:,::1],int64[:,::1],float64[:,::1],float64[:,:,::1],int64[:,::1],float64[::1],int64)")
def A_assembly_3D(SC_mat_e, ele_id, init_pos, PQ_detJ_e, A_RC, A_data, p_num):  
    count_sparse = 0
    TT_E = len(ele_id)

    for ele in range(TT_E):
        nodes = ele_id[ele]
        P_e = PQ_detJ_e[ele,0]
        Q_e = PQ_detJ_e[ele,1]
        R_e = PQ_detJ_e[ele,2]
        J_e = PQ_detJ_e[ele,3]

        SC_mat = SC_mat_e[ele]
        for i in range(4):
            row = nodes[i]
            c1, c2, c3, c4 = SC_mat[:, i]
            for j in range(4):
                col = nodes[j]
                NNx = 0
                NNy = 0
                NNz = 0
                for k in range(4):
                    P = P_e[k]
                    Q = Q_e[k]
                    R = R_e[k]
                    J = J_e[k]
                    Ni_with_gp = c1 + c2*P + c3*Q + c4*R
                    Nxj_with_gp = SC_mat[1, j]
                    Nyj_with_gp = SC_mat[2, j]
                    Nzj_with_gp = SC_mat[3, j]
                    NNx += 1/4 * Ni_with_gp * Nxj_with_gp * J
                    NNy += 1/4 * Ni_with_gp * Nyj_with_gp * J
                    NNz += 1/4 * Ni_with_gp * Nzj_with_gp * J

                A_RC[0, count_sparse] = row
                A_RC[1, count_sparse] = col
                A_data[count_sparse] = NNx
                count_sparse += 1
                A_RC[0, count_sparse] = row + 1 * p_num
                A_RC[1, count_sparse] = col + 1 * p_num
                A_data[count_sparse] = NNy
                count_sparse += 1
                A_RC[0, count_sparse] = row + 2 * p_num
                A_RC[1, count_sparse] = col + 2 * p_num
                A_data[count_sparse] = NNz
                count_sparse += 1
                A_RC[0, count_sparse] = row + 3 * p_num
                A_RC[1, count_sparse] = col + 3 * p_num
                A_data[count_sparse] = NNy
                count_sparse += 1
                A_RC[0, count_sparse] = row + 4 * p_num
                A_RC[1, count_sparse] = col + 4 * p_num
                A_data[count_sparse] = NNz
                count_sparse += 1
                A_RC[0, count_sparse] = row + 5 * p_num
                A_RC[1, count_sparse] = col + 5 * p_num
                A_data[count_sparse] = NNx
                count_sparse += 1
                A_RC[0, count_sparse] = row + 6 * p_num
                A_RC[1, count_sparse] = col + 6 * p_num
                A_data[count_sparse] = NNz
                count_sparse += 1
                A_RC[0, count_sparse] = row + 7 * p_num
                A_RC[1, count_sparse] = col + 7 * p_num
                A_data[count_sparse] = NNx
                count_sparse += 1
                A_RC[0, count_sparse] = row + 8 * p_num
                A_RC[1, count_sparse] = col + 8 * p_num
                A_data[count_sparse] = NNy
                count_sparse += 1

#@njit("void(float64[:,:,::1],int64[:,::1],float64[:,::1],float64[:,:,::1],float64[::1],int64)")
def R_assembly_3D(SC_mat_e, ele_id, init_pos, PQ_detJ_e, R_vec, p_num):
    TT_E = len(ele_id)

    for ele in range(TT_E):
        nodes = ele_id[ele]
        P_e = PQ_detJ_e[ele,0]
        Q_e = PQ_detJ_e[ele,1]
        R_e = PQ_detJ_e[ele,2]
        J_e = PQ_detJ_e[ele,3]

        SC_mat = SC_mat_e[ele]
        for i in range(4): 
            row = nodes[i]
            c1, c2, c3, c4 = SC_mat[:, i]
            N = 0
            for k in range(4):
                P = P_e[k]
                Q = Q_e[k]
                R = R_e[k]
                J = J_e[k]
                Ni_with_gp = c1 + c2*P + c3*Q + c4*R
                N += 1/4 * Ni_with_gp * J
            # R_vec bakes the reference "+1" into the diagonal components, so
            # those blocks come out as the deformation gradient (F_ii = 1 + H_ii)
            # rather than the displacement gradient. A_assembly_3D emits the
            # blocks in the order H11, H22, H33, H12, H23, H31, H13, H21, H32
            # (diagonal first, as in the 2D reference Assembly.py), so the
            # diagonal lives in blocks 0, 1, 2 -- verified by solving a known
            # linear displacement field u = H X and reading back each block.
            R_vec[row] += N
            R_vec[row + p_num] += N
            R_vec[row + 2 * p_num] += N

# ---------------------------------------------------------------------------
# Block-structured assembly.
#
# The full 9*p_num system built above is block diagonal, and its blocks are
# highly redundant (verified numerically, see README):
#   - all nine diagonal blocks of M are the SAME p_num x p_num mass matrix
#   - A has only THREE distinct blocks (the x-, y- and z-derivative operators)
#   - R is the same p_num vector repeated on the three diagonal components
#
# So the functions below assemble one copy of each distinct block. Solving
# then means factorising a single p_num x p_num matrix and reusing it for
# nine right-hand sides, instead of factorising a 9*p_num system: ~6x less
# assembly memory and a far cheaper solve, with identical results.
# ---------------------------------------------------------------------------

def M_assembly_3D_block(SC_mat_e, ele_id, init_pos, PQ_detJ_e, M_RC, M_data):
    """The single p_num x p_num consistent mass matrix block."""
    count_sparse = 0
    for ele in range(len(ele_id)):
        nodes = ele_id[ele]
        P_e, Q_e, R_e, J_e = PQ_detJ_e[ele, 0], PQ_detJ_e[ele, 1], PQ_detJ_e[ele, 2], PQ_detJ_e[ele, 3]
        SC_mat = SC_mat_e[ele]
        for i in range(4):
            row = nodes[i]
            c1, c2, c3, c4 = SC_mat[:, i]
            for j in range(4):
                col = nodes[j]
                NN = 0
                for k in range(4):
                    P, Q, R, J = P_e[k], Q_e[k], R_e[k], J_e[k]
                    Ni = c1 + c2*P + c3*Q + c4*R
                    Nj = SC_mat[0, j] + SC_mat[1, j]*P + SC_mat[2, j]*Q + SC_mat[3, j]*R
                    NN += 1/4 * Ni * Nj * J
                M_RC[0, count_sparse] = row
                M_RC[1, count_sparse] = col
                M_data[count_sparse] = NN
                count_sparse += 1


def A_assembly_3D_block(SC_mat_e, ele_id, init_pos, PQ_detJ_e, A_RC, Ax_data, Ay_data, Az_data):
    """The three distinct p_num x p_num gradient blocks (d/dx, d/dy, d/dz).
    They share the same sparsity pattern, so one A_RC serves all three."""
    count_sparse = 0
    for ele in range(len(ele_id)):
        nodes = ele_id[ele]
        P_e, Q_e, R_e, J_e = PQ_detJ_e[ele, 0], PQ_detJ_e[ele, 1], PQ_detJ_e[ele, 2], PQ_detJ_e[ele, 3]
        SC_mat = SC_mat_e[ele]
        for i in range(4):
            row = nodes[i]
            c1, c2, c3, c4 = SC_mat[:, i]
            for j in range(4):
                col = nodes[j]
                NNx = NNy = NNz = 0
                for k in range(4):
                    P, Q, R, J = P_e[k], Q_e[k], R_e[k], J_e[k]
                    Ni = c1 + c2*P + c3*Q + c4*R
                    NNx += 1/4 * Ni * SC_mat[1, j] * J
                    NNy += 1/4 * Ni * SC_mat[2, j] * J
                    NNz += 1/4 * Ni * SC_mat[3, j] * J
                A_RC[0, count_sparse] = row
                A_RC[1, count_sparse] = col
                Ax_data[count_sparse] = NNx
                Ay_data[count_sparse] = NNy
                Az_data[count_sparse] = NNz
                count_sparse += 1


def R_assembly_3D_block(SC_mat_e, ele_id, init_pos, PQ_detJ_e, R_vec):
    """The single p_num reference vector (integral of N_i). Added to the
    right-hand side of the three diagonal components only."""
    for ele in range(len(ele_id)):
        nodes = ele_id[ele]
        P_e, Q_e, R_e, J_e = PQ_detJ_e[ele, 0], PQ_detJ_e[ele, 1], PQ_detJ_e[ele, 2], PQ_detJ_e[ele, 3]
        SC_mat = SC_mat_e[ele]
        for i in range(4):
            row = nodes[i]
            c1, c2, c3, c4 = SC_mat[:, i]
            N = 0
            for k in range(4):
                P, Q, R, J = P_e[k], Q_e[k], R_e[k], J_e[k]
                N += 1/4 * (c1 + c2*P + c3*Q + c4*R) * J
            R_vec[row] += N
