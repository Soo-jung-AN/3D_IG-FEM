
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
            # R_vec bakes the reference "+1" into the solved deformation
            # gradient's DIAGONAL terms only (F11, F22, F33 -- blocks 0, 4, 8
            # in the 9-block unpacking order F11,F12,F13,F21,F22,F23,F31,F32,F33
            # used in main.py), matching the 2D IG-FEM reference implementation
            # (Assembly.py's R_assembly, which targets its F11/F22 diagonal
            # blocks 0 and 1). The previous blocks 0,1,2 mixed one diagonal
            # term (F11) with two off-diagonal ones (F12, F13) and left F22/F33
            # uncorrected, leaving the solved F un-physical (see the vol/E fix
            # in main.py).
            R_vec[row] += N
            R_vec[row + 4 * p_num] += N
            R_vec[row + 8 * p_num] += N