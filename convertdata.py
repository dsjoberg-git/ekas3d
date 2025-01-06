# Convert data saved by scatt2d.py to a single h5-file. In particular,
# instead of having the A-matrix distributed as several functions, it
# is now collected in one matrix object which is quicker to load in
# the postprocess script.
#
# Daniel Sjöberg, 2025-01-06

import numpy as np
import h5py

# Open files and read some data structures
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

f_in = h5py.File('output.h5', 'r')
Nx = len(f_in['Function']['real_f']['-3'])

# Save data in new h5-file
f_out = h5py.File('input.h5', 'w')

f_out['b'] = b
f_out['fvec'] = fvec
f_out['S_ref'] = S_ref
f_out['S_dut'] = S_dut
f_out['epsr_mat'] = epsr_mat
f_out['epsr_defect'] = epsr_defect

f_out.create_dataset('cell_volumes', (Nx,), dtype=float)
f_out['cell_volumes'][:] = f_in['Function']['real_f']['-3'][:].squeeze()

f_out.create_dataset('epsr_ref', (Nx,), dtype=complex)
f_out['epsr_ref'][:] = f_in['Function']['real_f']['-2'][:].squeeze() + 1j*f_in['Function']['imag_f']['-2'][:].squeeze()

f_out.create_dataset('epsr_dut', (Nx,), dtype=complex)
f_out['epsr_dut'][:] = f_in['Function']['real_f']['-1'][:].squeeze() + 1j*f_in['Function']['imag_f']['-1'][:].squeeze()

f_out.create_dataset('A', (Nb, Nx), dtype=complex)
for n in range(Nb):
    f_out['A'][n,:] = f_in['Function']['real_f'][str(n)][:].squeeze() + 1j*f_in['Function']['imag_f'][str(n)][:].squeeze()

f_out.close()
f_in.close()

