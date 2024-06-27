# Script to test optimization in adjoint formulation in a waveguide
# setting.
#
# Daniel Sjöberg, 2024-06-27

from mpi4py import MPI
import numpy as np
import dolfinx, ufl
import gmsh
from scipy.constants import c as c0
from matplotlib import pyplot as plt
import spgl1
from scipy.sparse.linalg import LinearOperator

filename_mesh = 'tmp.msh'
filename_output = 'output.npz'
svd_threshold = 1e-3  # Threshold for SVD
tdim = 2              # Geometrical dimension
fdim = tdim - 1       # Facet dimension

# Load mesh and prepare function space for delta_epsr
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(filename_mesh, comm=MPI.COMM_WORLD, rank=0, gdim=tdim)
Wspace = dolfinx.fem.functionspace(mesh, ('DG', 0))
delta_epsr = dolfinx.fem.Function(Wspace)

# Load previously computed values
data = np.load(filename_output)
A0 = data['A0']
A1 = data['A1']
A2 = data['A2']
A3 = data['A3']
b0 = data['b0']
b1 = data['b1']
b2 = data['b2']
b3 = data['b3']
fvec = data['fvec']
S_ref = data['S_ref']
S_dut = data['S_dut']
epsr_mat = data['epsr_mat']
epsr_defect = data['epsr_defect']
epsr_array_ref = data['epsr_array_ref']
epsr_array_dut = data['epsr_array_dut']
cell_volumes = data['cell_volumes']

if False: # Reduce data to only selected combinations
    Nf = len(fvec)
    Np = S_ref.shape[-1]
    ports = np.arange(Np)
    idx = np.zeros((Nf - 1)*Np*Np + (Np - 1)*Np + Np - 1 + 1, dtype=bool)
    for nf in range(Nf):
        for m in range(Np):
            for n in range(Np):
                if m == n:
                    idx[nf*Np*Np + m*Np + n] = True
                else:
                    idx[nf*Np*Np + m*Np + n] = False
    A0 = A0[idx,:]
    b0 = b0[idx]
    A1 = A1[idx,:]
    b1 = b1[idx]
    A2 = A2[idx,:]
    b2 = b2[idx]
    A3 = A3[idx,:]
    b3 = b3[idx]

print('Computing TSVD permittivity')
A0_inv = np.linalg.pinv(A0, rcond=svd_threshold)
z0_tsvd = np.dot(A0_inv, b0)
del(A0_inv)
A1_inv = np.linalg.pinv(A1, rcond=svd_threshold)
z1_tsvd = np.dot(A1_inv, b1)
del(A1_inv)
A2_inv = np.linalg.pinv(A2, rcond=svd_threshold)
z2_tsvd = np.dot(A2_inv, b2)
del(A2_inv)
A3_inv = np.linalg.pinv(A3, rcond=svd_threshold)
z3_tsvd = np.dot(A3_inv, b3)
del(A3_inv)

print('Computing TSVD permittivity, a priori location')
idx = np.nonzero(epsr_array_ref > 1)[0]
A0_apriori = A0[:,idx]
A0_inv_apriori = np.linalg.pinv(A0_apriori, rcond=svd_threshold)
z0_tsvd_apriori = np.zeros(z0_tsvd.shape, dtype=complex)
z0_tsvd_apriori[idx] = np.dot(A0_inv_apriori, b0)
A1_apriori = A1[:,idx]
A1_inv_apriori = np.linalg.pinv(A1_apriori, rcond=svd_threshold)
z1_tsvd_apriori = np.zeros(z1_tsvd.shape, dtype=complex)
z1_tsvd_apriori[idx] = np.dot(A1_inv_apriori, b1)
A2_apriori = A2[:,idx]
A2_inv_apriori = np.linalg.pinv(A2_apriori, rcond=svd_threshold)
z2_tsvd_apriori = np.zeros(z2_tsvd.shape, dtype=complex)
z2_tsvd_apriori[idx] = np.dot(A2_inv_apriori, b2)
A3_apriori = A3[:,idx]
A3_inv_apriori = np.linalg.pinv(A3_apriori, rcond=svd_threshold)
z3_tsvd_apriori = np.zeros(z3_tsvd.shape, dtype=complex)
z3_tsvd_apriori[idx] = np.dot(A3_inv_apriori, b3)

# Save results and compute errors
np.savez('results.npz', z0_tsvd=z0_tsvd, z1_tsvd=z1_tsvd, z2_tsvd=z2_tsvd, z3_tsvd=z3_tsvd, z0_tsvd_apriori=z0_tsvd_apriori, z1_tsvd_apriori=z1_tsvd_apriori, z2_tsvd_apriori=z2_tsvd_apriori, z3_tsvd_apriori=z3_tsvd_apriori, cell_volumes=cell_volumes) 
def norm(diff_epsr_try, diff_epsr_true, cell_volumes, p=2):
    volume = np.sum(cell_volumes)
    error = (np.sum(np.abs(diff_epsr_try - diff_epsr_true)**p*cell_volumes/volume))**(1/p)
    return error
epsr_true = epsr_array_dut - epsr_array_ref
p = 2
error1 = norm(z0_tsvd, epsr_true, cell_volumes, p=p)
error2 = norm(z1_tsvd, epsr_true, cell_volumes, p=p)
error3 = norm(z2_tsvd, epsr_true, cell_volumes, p=p)
error4 = norm(z3_tsvd, epsr_true, cell_volumes, p=p)
error5 = norm(z0_tsvd_apriori[idx], epsr_true[idx], cell_volumes[idx], p=p)
error6 = norm(z1_tsvd_apriori[idx], epsr_true[idx], cell_volumes[idx], p=p)
error7 = norm(z2_tsvd_apriori[idx], epsr_true[idx], cell_volumes[idx], p=p)
error8 = norm(z3_tsvd_apriori[idx], epsr_true[idx], cell_volumes[idx], p=p)
print(f'Case 1: Eref*Edut: error = {error1}')
print(f'Case 2: Eref*Eref: error = {error2}')
print(f'Case 3: Eref*conj(Edut): error = {error3}')
print(f'Case 4: Eref*conj(Eref): error = {error4}')
print(f'Case 5: Eref*Edut known shape: error = {error5}')
print(f'Case 6: Eref*Eref known shape: error = {error6}')
print(f'Case 7: Eref*conj(Edut) known shape: error = {error7}')
print(f'Case 8: Eref*conj(Eref) known shape: error = {error8}')


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'epsr.xdmf', 'w') as f:
    f.write_mesh(mesh)
    delta_epsr.x.array[:] = epsr_array_dut + 0j
    f.write_function(delta_epsr, 0)
    delta_epsr.x.array[:] = z0_tsvd + 0j
    f.write_function(delta_epsr, 1)
    delta_epsr.x.array[:] = z1_tsvd + 0j
    f.write_function(delta_epsr, 2)
    delta_epsr.x.array[:] = z2_tsvd + 0j
    f.write_function(delta_epsr, 3)
    delta_epsr.x.array[:] = z3_tsvd + 0j
    f.write_function(delta_epsr, 4)
    delta_epsr.x.array[:] = z0_tsvd_apriori + 0j
    f.write_function(delta_epsr, 5)
    delta_epsr.x.array[:] = z1_tsvd_apriori + 0j
    f.write_function(delta_epsr, 6)
    delta_epsr.x.array[:] = z2_tsvd_apriori + 0j
    f.write_function(delta_epsr, 7)
    delta_epsr.x.array[:] = z3_tsvd_apriori + 0j
    f.write_function(delta_epsr, 8)
    
