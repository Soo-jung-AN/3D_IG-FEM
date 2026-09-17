"""The full strain tensor from a DEM run, and what it says about the run.

vol_strain() returns only det(F) - 1. This returns every component of F,
and derives from it the quantities that actually diagnose a deformation:

  E        Green-Lagrangian strain, E = 0.5 (F^T F - I)
  e1>=e2>=e3   principal strains, and the Flinn-type ratios
  gamma    maximum shear strain, (e1 - e3) / 2
  R        the rotation from the polar decomposition F = R U, and the
           vertical-axis rotation angle read off it -- the quantity that
           tells a strike-slip model apart from a pure-shear one
  vol      det(F) - 1

Usage:
  python3 strain_analysis.py --init init_pos.txt --pos pos.txt \
                             [--alpha 125] [--out out.npz]
"""
import argparse
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from Assembly3 import M_assembly_3D_block, A_assembly_3D_block, R_assembly_3D_block
from preprocessing3 import Get_shf_coef_3D, Get_gp_cood_3D
from vp_from_strain import build_mesh


def deformation_gradient(X0, X1, ele_id):
    """All nine components of F at every particle, by the IG-FEM
    projection. Returns an (n, 3, 3) array."""
    p_num, TT_E = len(X0), len(ele_id)
    U = X1 - X0

    SC = np.zeros((TT_E, 4, 4)); Get_shf_coef_3D(SC, ele_id, X0)
    PQ = np.zeros((TT_E, 4, 4)); Get_gp_cood_3D(PQ, ele_id, X0)

    nnz = 16 * TT_E
    M_RC = np.zeros((2, nnz), dtype=np.int64); M_data = np.zeros(nnz)
    M_assembly_3D_block(SC, ele_id, X0, PQ, M_RC, M_data)
    M0 = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num, p_num)).tocsc()
    del M_RC, M_data

    A_RC = np.zeros((2, nnz), dtype=np.int64)
    Ad = [np.zeros(nnz) for _ in range(3)]
    A_assembly_3D_block(SC, ele_id, X0, PQ, A_RC, *Ad)
    A = [sparse.csr_matrix((d, (A_RC[0], A_RC[1])), shape=(p_num, p_num)) for d in Ad]
    del A_RC, Ad

    R0 = np.zeros(p_num)
    R_assembly_3D_block(SC, ele_id, X0, PQ, R0)
    del SC, PQ

    lu = splu(M0)
    F = np.empty((p_num, 3, 3))
    for i in range(3):          # component of u
        for j in range(3):      # derivative direction
            rhs = A[j] @ U[:, i]
            if i == j:
                rhs = rhs + R0          # F_ii = 1 + du_i/dX_i
            F[:, i, j] = lu.solve(rhs)
    return F


def analyse(F):
    """Strain invariants and the rotation, from F."""
    I = np.eye(3)
    E = 0.5 * (np.einsum("nki,nkj->nij", F, F) - I)
    e = np.linalg.eigvalsh(E)[:, ::-1]            # e1 >= e2 >= e3
    vol = np.linalg.det(F) - 1.0

    # polar decomposition F = R U, via the SVD: R = W V^T
    W, _, Vt = np.linalg.svd(F)
    Rot = np.einsum("nij,njk->nik", W, Vt)
    # rotation angle about each axis, from the skew part of R
    ang = np.stack([Rot[:, 2, 1] - Rot[:, 1, 2],
                    Rot[:, 0, 2] - Rot[:, 2, 0],
                    Rot[:, 1, 0] - Rot[:, 0, 1]], axis=1) / 2.0
    theta = np.degrees(np.arcsin(np.clip(np.linalg.norm(ang, axis=1), -1, 1)))
    trace = np.clip((np.trace(Rot, axis1=1, axis2=2) - 1) / 2, -1, 1)
    theta_total = np.degrees(np.arccos(trace))
    # signed rotation about the vertical (z) axis
    theta_z = np.degrees(np.arctan2(Rot[:, 1, 0] - Rot[:, 0, 1],
                                    Rot[:, 0, 0] + Rot[:, 1, 1]))
    return dict(E=E, e1=e[:, 0], e2=e[:, 1], e3=e[:, 2], vol=vol,
                gamma=(e[:, 0] - e[:, 2]) / 2.0,
                theta=theta_total, theta_z=theta_z, axis=ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--q-min", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=None,
                    help="drop particles above this z before meshing")
    ap.add_argument("--out", default="./results/strain_tensor.npz")
    args = ap.parse_args()
    t0 = time.time()

    X0 = np.loadtxt(args.init)
    X1 = np.loadtxt(args.pos)
    if args.zmax is not None:
        keep = X0[:, 2] < args.zmax
        print(f"cutting {(~keep).sum()} particles above z = {args.zmax:.0f} m")
        X0, X1 = X0[keep], X1[keep]
    ele_id, _ = build_mesh(X0, args.q_min, args.alpha)
    F = deformation_gradient(X0, X1, ele_id)
    a = analyse(F)
    E = a["E"]

    ok = np.abs(a["vol"]) < 1           # exclude residual mesh outliers
    lab = ("E11", "E22", "E33", "E12", "E13", "E23")
    idx = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    print(f"\nGreen-Lagrangian strain over {ok.sum()} of {len(X0)} particles "
          f"(|det F - 1| < 1):")
    print(f"  {'':>5} {'mean':>10} {'median':>10} {'sd':>10} {'p1':>10} {'p99':>10}")
    for l, (i, j) in zip(lab, idx):
        v = E[ok, i, j]
        print(f"  {l:>5} {v.mean():+10.5f} {np.median(v):+10.5f} {v.std():10.5f}"
              f" {np.percentile(v,1):+10.5f} {np.percentile(v,99):+10.5f}")
    for k in ("e1", "e2", "e3", "gamma", "vol", "theta", "theta_z"):
        v = a[k][ok]
        print(f"  {k:>5} {v.mean():+10.5f} {np.median(v):+10.5f} {v.std():10.5f}"
              f" {np.percentile(v,1):+10.5f} {np.percentile(v,99):+10.5f}")

    np.savez_compressed(args.out, X0=X0, X1=X1, F=F.astype(np.float32),
                        **{k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                           for k, v in a.items()})
    print(f"\nsaved {args.out}   ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
