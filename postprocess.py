# Script for finding the permittivity of defects using data saved from
# scatt2d.py in different files.
#
# Daniel Sjöberg, 2024-12-13

from mpi4py import MPI
import numpy as np
import dolfinx, ufl
import gmsh
from scipy.constants import c as c0
from matplotlib import pyplot as plt
import spgl1
from scipy.sparse.linalg import LinearOperator
import h5py

svd_threshold = 1e-3  # Threshold for SVD
tdim = 2              # Geometrical dimension
fdim = tdim - 1       # Facet dimension

# Load previously computed values
data = np.load('output.npz')
b = data['b']
fvec = data['fvec']
S_ref = data['S_ref']
S_dut = data['S_dut']
epsr_mat = data['epsr_mat']
epsr_defect = data['epsr_defect']
Nf = len(fvec)
Np = S_ref.shape[-1]
Nb = len(b)

# Load mesh from xdmf file and prepare function space. Note that
# dolfinx reorders the mesh when read.
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'output.xdmf', 'r') as f:
    mesh = f.read_mesh()
Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
delta_epsr = dolfinx.fem.Function(Wspace)

# Load values from h5 file and reorder them to the dolfinx mesh
idx = mesh.topology.original_cell_index
with h5py.File('output.h5', 'r') as f:
    cell_volumes = np.array(f['Function']['real_f']['-3']).squeeze()
    cell_volumes[:] = cell_volumes[idx]
    epsr_array_ref = np.array(f['Function']['real_f']['-2']).squeeze() + 1j*np.array(f['Function']['imag_f']['-2']).squeeze()
    epsr_array_ref = epsr_array_ref[idx]
    epsr_array_dut = np.array(f['Function']['real_f']['-1']).squeeze() + 1j*np.array(f['Function']['imag_f']['-1']).squeeze()
    epsr_array_dut = epsr_array_dut[idx]
    N = len(cell_volumes)
    A = np.zeros((Nb, N), dtype=complex)
    for n in range(Nb):
        A[n,:] = np.array(f['Function']['real_f'][str(n)]).squeeze() + 1j*np.array(f['Function']['imag_f'][str(n)]).squeeze()
        A[n,:] = A[n,idx]

if False: # Reduce data to only selected combinations
    idx = np.zeros((Nf - 1)*Np*Np + (Np - 1)*Np + Np - 1 + 1, dtype=bool)
    for nf in range(Nf):
        for m in range(Np):
            for n in range(Np):
                if m == n:
                    idx[nf*Np*Np + m*Np + n] = True
                else:
                    idx[nf*Np*Np + m*Np + n] = False
    A = A[idx,:]
    b = b[idx]

print('Computing TSVD permittivity')
A_inv = np.linalg.pinv(A, rcond=svd_threshold)
z_tsvd = np.dot(A_inv, b)
del(A_inv)

print('Computing TSVD permittivity, a priori location')
idx = np.nonzero(np.abs(epsr_array_ref) > 1)[0]
A_apriori = A[:,idx]
A_inv_apriori = np.linalg.pinv(A_apriori, rcond=svd_threshold)
z_tsvd_apriori = np.zeros(z_tsvd.shape, dtype=complex)
z_tsvd_apriori[idx] = np.dot(A_inv_apriori, b)

# Save results and compute errors
np.savez('results.npz', z_tsvd=z_tsvd, z_tsvd_apriori=z_tsvd_apriori, cell_volumes=cell_volumes) 
def norm(diff_epsr_try, diff_epsr_true, cell_volumes, p=2):
    volume = np.sum(cell_volumes)
    error = (np.sum(np.abs(diff_epsr_try - diff_epsr_true)**p*cell_volumes/volume))**(1/p)
    return error
epsr_true = epsr_array_dut - epsr_array_ref
p = 2
error1 = norm(z_tsvd, epsr_true, cell_volumes, p=p)
error2 = norm(z_tsvd_apriori[idx], epsr_true[idx], cell_volumes[idx], p=p)
print(f'Case Eref*Eref: error = {error1}')
print(f'Case Eref*Eref known shape: error = {error2}')

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'epsr.xdmf', 'w') as f:
    f.write_mesh(mesh)
    delta_epsr.x.array[:] = epsr_array_dut + 0j
    f.write_function(delta_epsr, 0)
    delta_epsr.x.array[:] = z_tsvd + 0j
    f.write_function(delta_epsr, 1)
    delta_epsr.x.array[:] = z_tsvd_apriori + 0j
    f.write_function(delta_epsr, 2)
    
