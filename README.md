# 3D_IG-FEM
3-dimensional IG-FEM

<img width="1353" height="1250" alt="Image" src="https://github.com/user-attachments/assets/71644e9b-10cc-46df-9956-970df1b96dbe" />




3D Delanauy triangulation is required to generate vtu mesh domain (e.g., tetrahedrone.vtu in the "main.py") using an open source visualization software, Paraview (https://www.paraview.org/).

3D mass matrix inversion of sparse linear system is conducted via Intel MKL Pardiso direct solver library, PyPardiso (https://pypi.org/project/pypardiso/).

Please install "pypardiso" and "vtk" following code in terminal before running the "main.py"

"pip install pypardiso"

"pip install vtk"

# Memory & Runtime overview

Total discrete element particles         : 259,943

Total triangular mesh elements           : 1,550,208

RAM usages (Max)                         : 17 GB for solving (only CPU-used), 35 GB for creating resulting visulaization file (.vtk)

Elasped time (IG-FEM solving)            : 349.10 sec 

Elasped time (VTKUnstructuredConverter2) : 350.70 sec


Strain analysis were conducted on a Linux workstation equipped with a 32-core Intel Xeon CPU, 128 GB RAM, and an NVIDIA RTX GPU Ada6000 running Ubuntu 22.04.
