# 3D_IG-FEM
3-dimensional IG-FEM

<img width="1353" height="1250" alt="Image" src="https://github.com/user-attachments/assets/71644e9b-10cc-46df-9956-970df1b96dbe" />




3D Delanauy triangulation is required to generate vtu mesh domain (e.g., tetrahedrone.vtu in the "main.py") using an open source visualization software, Paraview (https://www.paraview.org/).

The 9*p_num sparse system is block diagonal: all nine of its mass-matrix
blocks are the same p_num x p_num matrix, and the gradient operator has only
three distinct blocks (d/dx, d/dy, d/dz). main.py assembles one copy of each
and reuses a single LU factorisation for the nine right-hand sides, so the
solve is p_num-sized rather than 9*p_num-sized. That is small enough for
scipy's sparse solver; PyPardiso is no longer required.

Please install "vtk" following code in terminal before running the "main.py"

"pip install vtk"

(numpy, scipy and matplotlib are also required.)

# Memory & Runtime overview

Total discrete element particles         : 259,943

Total tetrahedral mesh elements          : 1,550,208 (1,469,046 after the
                                           sliver filter, see below)

RAM usage (peak)                         : 8.5 GB, whole pipeline including
                                           the .vtk export

Elapsed time (IG-FEM solving)            : 967 sec

Elapsed time (VTKUnstructuredConverter2) : 6 sec

The assembly loops are plain Python (the numba decorators in Assembly3.py /
preprocessing3.py are commented out), so the ~16 min is dominated by the
element loop, not by the linear algebra. Re-enabling numba would cut it
substantially.

Earlier versions of this code needed ~17 GB to solve and ~35 GB to write the
.vtk, and were run on a 32-core Xeon workstation with 128 GB RAM. Both
figures were artefacts rather than requirements: the solve was assembling
nine redundant copies of the same system, and the writer buffered the entire
output file as a list of Python strings before writing it. The numbers above
were measured on an ordinary 4-core, 15 GB container.

# Sliver filtering

reshape_3D drops degenerate tetrahedra by the scale-invariant shape quality
q = 6*sqrt(2)*V / L_max^3 (1 for a regular tetrahedron, 0 for a sliver),
with a default threshold of q_min = 0.05. The previous test was on absolute
volume (< 1e-6), which is scale dependent and removed 0 of the 1,550,208
elements of this mesh even though ~5% of them are slivers. Those slivers are
where the recovered deformation gradient blows up: without the filter, vol
ranged from -392 to +211 against a median of -0.07. See the docstring in
preprocessing3.py for the threshold sweep.
