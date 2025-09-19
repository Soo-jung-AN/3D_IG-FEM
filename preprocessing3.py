
import numpy.linalg as lina
import numpy as np
#from numba import njit, float64, int64, jit


#def reshape(init_pos, ele_id, alpha):
    #del_id = []
    # for i in range(len(ele_id)):
    #     id_tri_temp = ele_id[i]
    #     a,b,c = init_pos[id_tri_temp]
    #     length1 = np.sqrt(np.sum((a-b)**2))
    #     length2 = np.sqrt(np.sum((a-c)**2))
    #     length3 = np.sqrt(np.sum((b-c)**2))
    #     if length1 > alpha or length2 > alpha or length3 > alpha:
    #         del_id.append(i)
    # return np.delete(ele_id,del_id,0)

def reshape_3D(undeformed_cood, ele_id):
    del_id = []
    for i in range(len(ele_id)):
        e1, e2, e3, e4 = ele_id[i]
        x1, y1, z1 = undeformed_cood[e1]
        x2, y2, z2 = undeformed_cood[e2]
        x3, y3, z3 = undeformed_cood[e3]
        x4, y4, z4 = undeformed_cood[e4]
        
        # Tetrahedral Volume 계산
        v_matrix = np.array([
            [x2-x1, x3-x1, x4-x1],
            [y2-y1, y3-y1, y4-y1],
            [z2-z1, z3-z1, z4-z1]
        ])
        volume = np.abs(np.linalg.det(v_matrix)) / 6.0
        
        if volume < 1e-6:
            del_id.append(i)
    return np.delete(ele_id, del_id, 0)

def remove_unused_nodes(undeformed_cood, ele_id):
    used_nodes = np.unique(ele_id).astype(np.int64)  
    new_cood = undeformed_cood[used_nodes]
    mapping = {old: new for new, old in enumerate(used_nodes)}
    new_ele_id = np.vectorize(mapping.get)(ele_id)
    return new_cood, new_ele_id

#@njit("void(float64[:,:,::1], int64[:,::1], float64[:,::1])")
def Get_shf_coef_3D(SC_mat_e, ele_id, init_pos):
    Base4x4 = np.zeros((4,4), dtype=np.float64)
    for ele in range(len(ele_id)):
        n1, n2, n3, n4 = ele_id[ele]
        x1, x2, x3, x4 = init_pos[n1,0], init_pos[n2,0], init_pos[n3,0], init_pos[n4,0]
        y1, y2, y3, y4 = init_pos[n1,1], init_pos[n2,1], init_pos[n3,1], init_pos[n4,1]
        z1, z2, z3, z4 = init_pos[n1,2], init_pos[n2,2], init_pos[n3,2], init_pos[n4,2]

        Base4x4[0, :] = [1, x1, y1, z1]
        Base4x4[1, :] = [1, x2, y2, z2]
        Base4x4[2, :] = [1, x3, y3, z3]
        Base4x4[3, :] = [1, x4, y4, z4]
        
        SC_mat = np.linalg.inv(Base4x4)
        SC_mat_e[ele, :, :] = SC_mat

#@njit("void(float64[:,:,::1],int64[:,::1],float64[:,::1])")
def Get_gp_cood_3D(PQ_detJ_e, ele_id, init_pos):
    s_list = np.array([0.58541020, 0.13819660, 0.13819660, 0.13819660], dtype=np.float64)
    t_list = np.array([0.13819660, 0.58541020, 0.13819660, 0.13819660], dtype=np.float64)
    r_list = np.array([0.13819660, 0.13819660, 0.58541020, 0.13819660], dtype=np.float64)

    for ele in range(len(ele_id)):
        n1, n2, n3, n4 = ele_id[ele]
        x1, x2, x3, x4 = init_pos[n1,0], init_pos[n2,0], init_pos[n3,0], init_pos[n4,0]
        y1, y2, y3, y4 = init_pos[n1,1], init_pos[n2,1], init_pos[n3,1], init_pos[n4,1]
        z1, z2, z3, z4 = init_pos[n1,2], init_pos[n2,2], init_pos[n3,2], init_pos[n4,2]

        for i in range(4):
            s, t, r = s_list[i], t_list[i], r_list[i]
            N1, N2, N3, N4 = 1 - s - t - r, s, t, r
            P = x1 * N1 + x2 * N2 + x3 * N3 + x4 * N4
            Q = y1 * N1 + y2 * N2 + y3 * N3 + y4 * N4
            R = z1 * N1 + z2 * N2 + z3 * N3 + z4 * N4

            PQ_detJ_e[ele,0,i] = P
            PQ_detJ_e[ele,1,i] = Q
            PQ_detJ_e[ele,2,i] = R

            J_matrix = np.array([
                [x2 - x1, x3 - x1, x4 - x1],
                [y2 - y1, y3 - y1, y4 - y1],
                [z2 - z1, z3 - z1, z4 - z1]
            ])
            det_J = np.linalg.det(J_matrix)
            PQ_detJ_e[ele,3,i] = det_J
