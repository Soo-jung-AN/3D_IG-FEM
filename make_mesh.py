"""Tetrahedral mesh from DEM particle positions, without ParaView.

The manual step this replaces: export the pack as a point .vtk, open it
in ParaView, apply the Delaunay3D filter, save the result as .vtu.

The obvious replacement is vtkDelaunay3D, which is the filter ParaView
runs, and vtk is already a dependency here. DO NOT USE IT on a DEM pack.
Measured on txt/init_pos.txt (259,943 particles):

    vtkDelaunay3D, offset 2.5     1,378,544 tets    37,607 orphans
    vtkDelaunay3D, offset 1000    1,383,409 tets    37,685 orphans
    scipy (Qhull)                 1,555,616 tets         0 orphans
    the ParaView mesh in txt/     1,550,208 tets         0 orphans

vtkDelaunay3D drops 14.5% of the points -- it warns about "degenerate
triangles" and gives up on them -- and those particles then have no
supporting element, which makes the IG-FEM mass matrix singular. Neither
Tolerance nor Offset changes this; a near-regular sphere packing is full
of co-spherical degeneracies that its incremental algorithm mishandles.

Qhull, through scipy.spatial.Delaunay, loses nothing and lands within
0.35% of the tetrahedron count of the ParaView mesh this repository has
always used. So scipy is the default here and --engine vtk is kept only
for comparison.

One thing Qhull needs that vtkDelaunay3D does not: its simplices come
back in arbitrary orientation (50% negative volume on this pack), and
the element integrals in Assembly3.py are weighted by det(J) with no
absolute value, so negative-volume elements subtract mass and the
recovered deformation gradient is garbage. Every tetrahedron is
re-oriented to positive volume before it is written.

Two further things ParaView does not do for you, both of which this
script can:

  --alpha-edge   drop tetrahedra with any edge longer than this. A
                 Delaunay triangulation fills the CONVEX HULL, so a pack
                 with a free surface gets long thin elements bridging the
                 empty space. (vtkDelaunay3D's own Alpha is a
                 circumsphere-radius test, not an edge-length one, and it
                 also emits triangles, lines and vertices, so this uses
                 an explicit edge filter instead.)
  --q-min        drop slivers on the scale-invariant shape quality
                 q = 6*sqrt(2)*V / L_max^3.

Both default to 0, i.e. off, so that the output matches ParaView's
unless you ask otherwise. main.py applies its own q filter on load.

Usage:
  python3 make_mesh.py positions.txt out.vtu [--alpha-edge 125] [--q-min 0.05]
  python3 make_mesh.py cloud.vtk out.vtu          # a point .vtk also works
  python3 make_mesh.py in.txt out.vtu --engine vtk   # the ParaView filter
"""
import argparse
import time

import numpy as np
from scipy.spatial import Delaunay

from preprocessing3 import tet_quality
from vp_from_strain import orient_positive


def _vtk():
    import vtk
    from vtk.util import numpy_support
    return vtk, numpy_support


def read_points(path):
    """Particle centres from a whitespace .txt or any VTK dataset."""
    if path.lower().endswith((".vtk", ".vtu", ".vtp")):
        vtk, numpy_support = _vtk()
        reader = {".vtk": vtk.vtkGenericDataObjectReader,
                  ".vtu": vtk.vtkXMLUnstructuredGridReader,
                  ".vtp": vtk.vtkXMLPolyDataReader}[path[-4:].lower()]()
        reader.SetFileName(path)
        reader.Update()
        out = reader.GetOutput()
        return numpy_support.vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    return np.loadtxt(path)


def delaunay_qhull(X):
    """scipy.spatial.Delaunay (Qhull), re-oriented. The default engine."""
    tets, n_flipped = orient_positive(X, Delaunay(X).simplices.astype(np.int64))
    print(f"  Qhull: {len(tets)} tetrahedra, {n_flipped} re-oriented to "
          f"positive volume", flush=True)
    return tets


def delaunay3d(X, tolerance=0.001, offset=2.5):
    """vtkDelaunay3D with ParaView's GUI defaults. See the module
    docstring: this loses 14.5% of the points on a DEM pack."""
    vtk, numpy_support = _vtk()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(X, dtype=float)))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)

    d = vtk.vtkDelaunay3D()
    d.SetInputData(poly)
    d.SetAlpha(0.0)
    d.SetTolerance(tolerance)
    d.SetOffset(offset)
    d.BoundingTriangulationOff()
    d.Update()
    grid = d.GetOutput()

    cells = numpy_support.vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    types = numpy_support.vtk_to_numpy(grid.GetCellTypes()
                                       if hasattr(grid, "GetCellTypes")
                                       else grid.GetCellTypesArray())
    if not np.all(types == vtk.VTK_TETRA):
        raise RuntimeError(f"expected only tetrahedra, got cell types "
                           f"{sorted(set(types.tolist()))}")
    return cells.reshape(-1, 4).astype(np.int64)


def max_edge(X, tets):
    a, b, c, d = (X[tets[:, k]] for k in range(4))
    return np.stack([np.linalg.norm(p - q, axis=1) for p, q in
                     ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))], axis=1).max(axis=1)


def write_vtu(path, X, tets):
    vtk, numpy_support = _vtk()
    grid = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(X, dtype=float)))
    grid.SetPoints(pts)

    cells = vtk.vtkCellArray()
    offs = numpy_support.numpy_to_vtkIdTypeArray(
        np.ascontiguousarray(np.arange(len(tets) + 1, dtype=np.int64) * 4), deep=1)
    conn_arr = numpy_support.numpy_to_vtkIdTypeArray(
        np.ascontiguousarray(tets.ravel()), deep=1)
    cells.SetData(offs, conn_arr)
    grid.SetCells(vtk.VTK_TETRA, cells)

    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(path)
    w.SetInputData(grid)
    w.Write()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("out")
    ap.add_argument("--alpha-edge", type=float, default=0.0)
    ap.add_argument("--q-min", type=float, default=0.0)
    ap.add_argument("--engine", choices=("qhull", "vtk"), default="qhull",
                    help="qhull (default, loses no points) or vtk (ParaView's "
                         "filter, which drops 14.5%% of a DEM pack)")
    a = ap.parse_args()

    t0 = time.time()
    X = read_points(a.positions)
    print(f"{len(X)} points from {a.positions}", flush=True)
    tets = delaunay_qhull(X) if a.engine == "qhull" else delaunay3d(X)
    print(f"  {len(tets)} tetrahedra   [{time.time()-t0:.0f} s]", flush=True)

    keep = np.ones(len(tets), bool)
    if a.alpha_edge > 0:
        long_edge = max_edge(X, tets) > a.alpha_edge
        keep &= ~long_edge
        print(f"  {100*long_edge.mean():.2f}% over alpha-edge = {a.alpha_edge:.0f}")
    if a.q_min > 0:
        sliver = tet_quality(X, tets) < a.q_min
        keep &= ~sliver
        print(f"  {100*sliver.mean():.2f}% slivers (q < {a.q_min})")
    tets = tets[keep]

    orphan = np.setdiff1d(np.arange(len(X)), np.unique(tets))
    if len(orphan):
        print(f"  WARNING: {len(orphan)} points have no supporting tetrahedron")

    v = np.einsum("ij,ij->i", X[tets[:, 1]] - X[tets[:, 0]],
                  np.cross(X[tets[:, 2]] - X[tets[:, 0]], X[tets[:, 3]] - X[tets[:, 0]]))
    print(f"  negative-volume tetrahedra: {100*np.mean(v < 0):.2f}%")

    write_vtu(a.out, X, tets)
    print(f"  wrote {a.out}: {len(tets)} tetrahedra   [{time.time()-t0:.0f} s total]")


if __name__ == "__main__":
    main()
