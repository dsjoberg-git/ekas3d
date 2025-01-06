# Script for finding the permittivity of defects using data saved from
# scatt2d.py. 
#
# Daniel Sjöberg, 2025-01-06

# These lines enable setting PETSc options on the command line. Can be
# used to have more information from the solver, for instance by
# "python postprocess_lsqr.py -ksp_view"
import sys, petsc4py
petsc4py.init(sys.argv)

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import dolfinx
import h5py
import time

rtol = 1e-3           # Relative tolerance for LSQR solution
tdim = 2              # Geometrical dimension
fdim = tdim - 1       # Facet dimension

# Set communicator and print function
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
model_rank = 0
Print = PETSc.Sys.Print

# Open h5-file, read some inputs and obtain lengths of vectors
f = h5py.File('input.h5', 'r')
Nx = len(f['cell_volumes'])
b0 = f['b']
fvec = f['fvec']
S_ref = f['S_ref']
S_dut = f['S_dut']
epsr_mat = f['epsr_mat']
epsr_defect = f['epsr_defect']
Nf = len(fvec)
Np = S_ref.shape[-1]
Nb = len(b0)

remove_data = False
if False:
    # Reduce data to only selected combinations. The current
    # implementation works, but reading the data is slow in parallel,
    # since the h5-file has to be read many times.
    remove_data = True
    keep_idx_b = []
    for nf in range(Nf):
        for m in range(Np):
            for n in range(Np):
                if m == n:
                    keep_idx_b.append(nf*Np*Np + m*Np + n)
    keep_idx_b = np.array(keep_idx_b, dtype=PETSc.IntType)
    Nb = len(keep_idx_b)

# Load mesh from xdmf file and prepare function space. Note that
# dolfinx reorders the mesh when read, this is indexed by idx0. This
# has to be sorted since the h5-file must be read with monotone list
# of indices. When running in parallel, idx0 as read from the mesh
# contains references to ghost cells, which are truncated by the
# "[:nlocal]" syntax.
with dolfinx.io.XDMFFile(comm, 'output.xdmf', 'r') as xdmf:
    mesh = xdmf.read_mesh()
Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
epsr = dolfinx.fem.Function(Wspace)
rstart, rend = epsr.x.petsc_vec.getOwnershipRange()
nlocal = epsr.x.petsc_vec.getLocalSize()
global_idx = np.array(mesh.topology.original_cell_index, dtype=PETSc.IntType)[:nlocal]
local_idx = PETSc.IntType(np.argsort(global_idx))

# Set up permittivity vectors based on epsr as read by xdmf
Print('Reading vectors')
tic = time.perf_counter()
x = epsr.x.petsc_vec.duplicate() # Solution vector
x_apriori = x.duplicate()        # Solution vector for a priori case
cell_volumes = x.duplicate()
cell_volumes[rstart+local_idx] = f['cell_volumes'][global_idx[local_idx]]
cell_volumes.assemble()
x_ref = x.duplicate()
x_ref[rstart+local_idx] = f['epsr_ref'][global_idx[local_idx]]
x_ref.assemble()
x_dut = x.duplicate()
x_dut[rstart+local_idx] = f['epsr_dut'][global_idx[local_idx]]
x_dut.assemble()
toc = time.perf_counter()
Print(f'Time for reading vectors: {toc-tic}')

# Set up matrix
A = PETSc.Mat().createDense([[Nb, Nb], [nlocal, Nx]], comm=comm)
A.setFromOptions()
A.setUp()

# Set up right hand side vector based on matrix A
b = A.getVecLeft()
if not remove_data:
    b[:] = b0[:Nb]
else:
    b[:] = b0[keep_idx_b]
b.assemble()

# Fill matrix
Print('Reading matrix')
tic = time.perf_counter()
if not remove_data:
    A[:Nb,rstart+local_idx] = f['A'][:Nb,global_idx[local_idx]]
else:
    for n, k in enumerate(keep_idx_b):
        A[n,rstart+local_idx] = f['A'][k,global_idx[local_idx]]
toc = time.perf_counter()
Print(f'Time for reading matrix: {toc-tic}')
tic = time.perf_counter()
A.assemble()
toc = time.perf_counter()
Print(f'Time for assembling matrix: {toc-tic}')
f.close()

# Set up PETSc LSQR solver
ksp = PETSc.KSP().create()
ksp.setOperators(A)
pc = ksp.getPC()
pc.setType('none')
ksp.setType('lsqr')
ksp.setTolerances(rtol=rtol)
ksp.setFromOptions()

Print('Computing LSQR permittivity')
tic = time.perf_counter()
ksp.solve(b, x)
toc = time.perf_counter()
Print(f'Solution time {toc-tic}')

# Constrain problem to non-air regions and solve again. Current
# implementation simply sets relevant columns of A to zero, it is
# probably better to create a new matrix with these columns removed,
# but I have not worked out how to do this using PETSc matrices.
Print('Reassembling matrix for a priori case')
idx = PETSc.IntType(np.nonzero(np.isclose(x_ref, 1))[0])
A[:,rstart+idx] = np.zeros((Nb, len(idx)), dtype=PETSc.ScalarType)
A.assemble()
Print('Computing LSQR permittivity - a priori case')
tic = time.perf_counter()
ksp.solve(b, x_apriori)
toc = time.perf_counter()
Print(f'Solution time {toc-tic}')

# Compute errors
global_volume = cell_volumes.sum()
error1 = ((x - (x_dut - x_ref))*cell_volumes/global_volume).norm(PETSc.NormType.NORM_2)
cell_volumes[idx] = np.zeros(len(idx), dtype=PETSc.ScalarType)
cell_volumes.assemble()
global_volume_apriori = cell_volumes.sum()
error2 = ((x_apriori - (x_dut - x_ref))*cell_volumes/global_volume).norm(PETSc.NormType.NORM_2)
Print(f'Case Eref*Eref: error1 = {error1}, error2 = {error2}')

# Save data
with dolfinx.io.XDMFFile(comm, 'epsr.xdmf', 'w') as f:
    f.write_mesh(mesh)
    epsr.x.petsc_vec[rstart:rend] = x_dut[rstart:rend]
    f.write_function(epsr, 0)
    epsr.x.petsc_vec[rstart:rend] = x[rstart:rend]
    f.write_function(epsr, 1)
    epsr.x.petsc_vec[rstart:rend] = x_apriori[rstart:rend]
    f.write_function(epsr, 2)
    
