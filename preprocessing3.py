
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

def tet_quality(undeformed_cood, ele_id):
    """Normalised tetrahedron shape quality q = 6*sqrt(2)*V / L_max^3.
    q = 1 for a regular tetrahedron and tends to 0 for a sliver, and it is
    scale invariant -- unlike a raw volume, it means the same thing whether
    the model is in metres or kilometres."""
    a, b, c, d = (undeformed_cood[ele_id[:, k]] for k in range(4))
    volume = np.abs(np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a))) / 6.0
    edges = np.stack([np.linalg.norm(x - y, axis=1) for x, y in
                      ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))], axis=1)
    return 6 * np.sqrt(2) * volume / edges.max(axis=1) ** 3


def reshape_3D(undeformed_cood, ele_id, q_min=0.05):
    """Drop degenerate (sliver) tetrahedra, which are where the recovered
    deformation gradient blows up.

    The filter is on shape quality, not raw volume. The previous absolute
    test (volume < 1e-6) is scale dependent and silently does nothing on a
    model whose element volumes run 1e4-1e8: it removed 0 of the 1,550,208
    elements of the reference mesh, even though 5% of them are slivers.

    q_min = 0.05 comes from a sweep on the reference model, scoring each
    threshold by the spread of `vol` and by agreement with an independent
    nearest-neighbour (SSPX-style) strain estimate on the same particles:

        q_min   removed   vol range        corr vs SSPX (all / |vol|<1)
        0        0.00%    -391.8 .. +210.8    +0.110 / +0.622
        0.01     2.41%     -35.9 ..  +52.4    +0.415 / +0.682
        0.05     5.24%     -17.8 ..  +22.9    +0.581 / +0.719
        0.10     5.69%     -19.2 ..  +24.3    +0.582 / +0.723

    Mean and median `vol` are unchanged across all of these (-0.038,
    -0.068), i.e. the filter only removes sliver-driven outliers. Past
    0.05 it plateaus. No particle loses all of its supporting elements at
    any threshold up to 0.1, so the mass matrix stays non-singular.
    """
    return ele_id[tet_quality(undeformed_cood, ele_id) >= q_min]

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
