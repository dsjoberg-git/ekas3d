# 2D scattering with open boundary implemented using stretched
# coordinate PML.
#
# Daniel Sjöberg, 2024-04-13

from mpi4py import MPI
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
import gmsh
from scipy.constants import c as c0
from matplotlib import pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt
import functools

# Physical constants and geometry dimension
mu0 = 4*np.pi*1e-7
eps0 = 1/c0**2/mu0
eta0 = np.sqrt(mu0/eps0)
tdim = 2
fdim = tdim - 1

# Simulation variables
f1 = 8e9                        # Start frequency
f2 = 12e9                       # Stop frequency
Nf = 10                         # Number of frequency points
fvec = np.linspace(f1, f2, Nf)  # Vector of simulation frequencies
f0 = 10e9                       # Design frequency
lambda0 = c0/f0                 # Design wavelength
k0 = 2*np.pi/lambda0            # Design wavenumber
h = lambda0/20                  # Mesh size
fem_degree = 1                  # Degree of finite elements
R_dom = 5*lambda0               # Radius of domain
d_pml = lambda0                 # Thickness of PML
R_pml = R_dom + d_pml           # Outer radius of PML

# Antennas
antenna_width = 0.7625*lambda0  # Width of antenna apertures, 22.86 mm
antenna_thickness = lambda0/10  # Thickness of antenna
kc = np.pi/antenna_width        # Cutoff wavenumber of antenna
N_antennas = 10                  # Number of antennas
R_antennas = 4*lambda0          # Radius at which antennas are placed
phi_antennas = np.linspace(0, 2*np.pi, N_antennas + 1)[:-1]
pos_antennas = np.array([[R_antennas*np.cos(phi), R_antennas*np.sin(phi), 0] for phi in phi_antennas])
rot_antennas = phi_antennas + np.pi/2

# Background, DUT, and defect
epsr_bkg = 1.0
mur_bkg = 1.0
mat_pos = np.array([0, 0, 0])
mat_width = lambda0
mat_height = lambda0
defect_pos = np.array([0, 0, 0])
defect_radius = 0.1*lambda0
epsr_mat = 3.0
epsr_defect = 2.5
mur_mat = 1.0
mur_defect = 1.0

# Set up mesh using gmsh
gmsh.initialize()
# Typical mesh size
gmsh.option.setNumber('General.Verbosity', 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

# Create antennas
antennas_DimTags = []
x_antenna = np.zeros((N_antennas, 3))
x_pec = np.zeros((N_antennas, 3, 3))
inAntennaSurface = []
inPECSurface = []
for n in range(N_antennas):
    rect = gmsh.model.occ.addRectangle(-antenna_width/2, -antenna_thickness, 0, antenna_width, antenna_thickness)
    gmsh.model.occ.rotate([(tdim, rect)], 0, 0, 0, 0, 0, 1, rot_antennas[n])
    gmsh.model.occ.translate([(tdim, rect)], pos_antennas[n,0], pos_antennas[n,1], pos_antennas[n,2])
    antennas_DimTags.append((tdim, rect))
    x_antenna[n] = pos_antennas[n, :]
    Rmat = np.array([[np.cos(rot_antennas[n]), -np.sin(rot_antennas[n]), 0],
                     [np.sin(rot_antennas[n]), np.cos(rot_antennas[n]), 0],
                     [0, 0, 1]])
    x_pec[n, 0] = x_antenna[n] + np.dot(Rmat, np.array([antenna_width/2, -antenna_thickness/2, 0]))
    x_pec[n, 1] = x_antenna[n] + np.dot(Rmat, np.array([-antenna_width/2, -antenna_thickness/2, 0]))
    x_pec[n, 2] = x_antenna[n] + np.dot(Rmat, np.array([0, -antenna_thickness, 0]))
    inAntennaSurface.append(lambda x: np.allclose(x, x_antenna[n]))
    inPECSurface.append(lambda x: np.allclose(x, x_pec[n,0]) or np.allclose(x, x_pec[n,1]) or np.allclose(x, x_pec[n,2]))

# Create material and defect
#mat = gmsh.model.occ.addRectangle(mat_pos[0] - mat_width/2, mat_pos[1] - mat_height/2, mat_pos[2], mat_width, mat_height)
#defect = gmsh.model.occ.addDisk(defect_pos[0], defect_pos[1], defect_pos[2], defect_radius, defect_radius)
if True:  # Aircraft geometry
    mat_body = gmsh.model.occ.addDisk(0, 0, 0, 3*lambda0, 0.5*lambda0)
    mat_tail = gmsh.model.occ.addDisk(0, 0, 0, lambda0, 0.3*lambda0)
    gmsh.model.occ.rotate([(tdim, mat_tail)], 0, 0, 0, 0, 0, 1, np.pi/2)
    gmsh.model.occ.translate([(tdim, mat_tail)], -2.5*lambda0, 0, 0)
    mat_wing1 = gmsh.model.occ.addDisk(0, 0, 0, 1.6*lambda0, 0.3*lambda0)
    mat_wing2 = gmsh.model.occ.addDisk(0, 0, 0, 1.6*lambda0, 0.3*lambda0)
    gmsh.model.occ.translate([(tdim, mat_wing1), (tdim, mat_wing2)], -1.1*lambda0, 0, 0)
    gmsh.model.occ.rotate([(tdim, mat_wing1)], 0, 0, 0, 0, 0, 1, 60*np.pi/180)
    gmsh.model.occ.rotate([(tdim, mat_wing2)], 0, 0, 0, 0, 0, 1, -60*np.pi/180)
    gmsh.model.occ.translate([(tdim, mat_wing1), (tdim, mat_wing2)], 0.6*lambda0, 0, 0)
    matDimTags, matDimTagsMap = gmsh.model.occ.fuse([(tdim, mat_body)], [(tdim, mat_tail), (tdim, mat_wing1), (tdim, mat_wing2)])

    defect1 = gmsh.model.occ.addDisk(0, lambda0, 0, defect_radius, defect_radius)
    defect2 = gmsh.model.occ.addDisk(-2*lambda0, 0, 0, defect_radius, defect_radius)
    defect3 = gmsh.model.occ.addRectangle(1.5*lambda0, -0.2*lambda0, 0, lambda0, 0.1*lambda0)
    defectDimTags, defectDimTagsMap = gmsh.model.occ.fuse([(tdim, defect1)], [(tdim, defect2), (tdim, defect3)])
else:  # Curved radome geometry
    mat_point1 = gmsh.model.occ.addPoint(-2.5*lambda0, 2*lambda0, 0)
    mat_point2 = gmsh.model.occ.addPoint(3*lambda0, 0, 0)
    mat_point3 = gmsh.model.occ.addPoint(-2.5*lambda0, -2*lambda0, 0)
    mat_point4 = gmsh.model.occ.addPoint(-2.5*lambda0, 1.5*lambda0, 0)
    mat_point5 = gmsh.model.occ.addPoint(2.5*lambda0, 0, 0)
    mat_point6 = gmsh.model.occ.addPoint(-2.5*lambda0, -1.5*lambda0, 0)
    mat_curve1 = gmsh.model.occ.addSpline([mat_point1, mat_point2, mat_point3])
    mat_curve2 = gmsh.model.occ.addSpline([mat_point4, mat_point5, mat_point6])
    mat_line1 = gmsh.model.occ.addLine(mat_point4, mat_point1)
    mat_line2 = gmsh.model.occ.addLine(mat_point3, mat_point6)
    mat_curveloop = gmsh.model.occ.addCurveLoop([mat_curve1, mat_line2, mat_curve2, mat_line1])
    mat = gmsh.model.occ.addPlaneSurface([mat_curveloop])
    matDimTags = [(tdim, mat)]

    defect1 = gmsh.model.occ.addDisk(-2.1*lambda0, 1.7*lambda0, 0, defect_radius, defect_radius)
    defect2 = gmsh.model.occ.addDisk(2.75*lambda0, 0, 0, defect_radius, defect_radius)
    defect3 = gmsh.model.occ.addRectangle(-1.7*lambda0, -1.7*lambda0, 0, lambda0, 0.1*lambda0)
    gmsh.model.occ.rotate([(tdim, defect3)], -1.7*lambda0, -1.7*lambda0, 0, 0, 0, 1, 11*np.pi/180)
    defectDimTags, defectDimTagsMap = gmsh.model.occ.fuse([(tdim, defect1)], [(tdim, defect2), (tdim, defect3)])

# Create domain and PML region
domain_disk = gmsh.model.occ.addDisk(0, 0, 0, R_dom, R_dom)
pml_disk = gmsh.model.occ.addDisk(0, 0, 0, R_pml, R_pml)

# Create fragments and dimtags
outDimTags, outDimTagsMap = gmsh.model.occ.fragment([(tdim, pml_disk)], [(tdim, domain_disk)] + matDimTags + defectDimTags + antennas_DimTags)
removeDimTags = [x for x in [y[0] for y in outDimTagsMap[-N_antennas:]]]
defectDimTags = [x[0] for x in outDimTagsMap[3:] if x[0] not in removeDimTags]
matDimTags = [x for x in outDimTagsMap[2] if x not in defectDimTags]
domainDimTags = [x for x in outDimTagsMap[1] if x not in removeDimTags+matDimTags+defectDimTags]
pmlDimTags = [x for x in outDimTagsMap[0] if x not in domainDimTags+defectDimTags+matDimTags+removeDimTags]
gmsh.model.occ.remove(removeDimTags)
gmsh.model.occ.synchronize()

# Make physical groups for domains and PML
mat_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in matDimTags])
defect_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in defectDimTags])
domain_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in domainDimTags])
pml_marker = gmsh.model.addPhysicalGroup(tdim, [x[1] for x in pmlDimTags])

# Identify antenna surfaces and make physical groups
pec_surface = []
antenna_surface = []
for boundary in gmsh.model.occ.getEntities(dim=fdim):
    CoM = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    for n in range(N_antennas):
        if inPECSurface[n](CoM):
            pec_surface.append(boundary[1])
        if inAntennaSurface[n](CoM):
            antenna_surface.append(boundary[1])
pec_surface_marker = gmsh.model.addPhysicalGroup(fdim, pec_surface)
antenna_surface_markers = [gmsh.model.addPhysicalGroup(fdim, [s]) for s in antenna_surface]
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(tdim)
gmsh.write('tmp.msh')
if False:
    gmsh.fltk.run()
    exit()
mesh, subdomains, boundaries = dolfinx.io.gmshio.model_to_mesh(
    gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=tdim)
gmsh.finalize()

mesh.topology.create_connectivity(tdim, tdim)

# Set up FEM function spaces and boundary conditions.
curl_element = basix.ufl.element('N1curl', mesh.basix_cell(), fem_degree)
lagrange_element = basix.ufl.element('CG', mesh.basix_cell(), fem_degree)
mixed_element = basix.ufl.mixed_element([curl_element, lagrange_element])
Vspace = dolfinx.fem.functionspace(mesh, mixed_element)
V0space, V0_dofs = Vspace.sub(0).collapse()
V1space, V1_dofs = Vspace.sub(1).collapse()

# Create measures for subdomains and surfaces
dx = ufl.Measure('dx', domain=mesh, subdomain_data=subdomains, metadata={'quadrature_degree': 5})
dx_dom = dx((domain_marker, mat_marker, defect_marker))
dx_pml = dx(pml_marker)
ds = ufl.Measure('ds', domain=mesh, subdomain_data=boundaries)
ds_antennas = [ds(m) for m in antenna_surface_markers]
ds_pec = ds(pec_surface_marker)

# Set up material parameters
Wspace = dolfinx.fem.functionspace(mesh, ("DG", 0))
epsr = dolfinx.fem.Function(Wspace)
mur = dolfinx.fem.Function(Wspace)
epsr.x.array[:] = epsr_bkg
mur.x.array[:] = mur_bkg
mat_cells = subdomains.find(mat_marker)
mat_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=mat_cells)
defect_cells = subdomains.find(defect_marker)
defect_dofs = dolfinx.fem.locate_dofs_topological(Wspace, entity_dim=tdim, entities=defect_cells)
epsr.x.array[mat_dofs] = epsr_mat
mur.x.array[mat_dofs] = mur_mat
epsr.x.array[defect_dofs] = epsr_mat
mur.x.array[defect_dofs] = mur_mat
epsr_array_ref = epsr.x.array.copy()
epsr.x.array[defect_dofs] = epsr_defect
mur.x.array[defect_dofs] = mur_defect
epsr_array_dut = epsr.x.array.copy()

# Set up PML layer
def pml_stretch(y, x, k0, x_dom=0, x_pml=1, n=3, R0=1e-10):
    return y*(1 - 1j*(n + 1)*np.log(1/R0)/(2*k0*np.abs(x_pml - x_dom))*((x - x_dom)/(x_pml - x_dom))**n)

def pml_epsr_murinv(pml_coords, r):
    J = ufl.grad(pml_coords)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    r_pml = ufl.sqrt(pml_coords[0]**2 + pml_coords[1]**2)
    J = ufl.as_matrix(((J[0, 0], J[0, 1], 0),
                       (J[1, 0], J[1, 1], 0),
                       (0, 0, r_pml / r)))
    A = ufl.inv(J)
    epsr_pml = ufl.det(J) * A * epsr * ufl.transpose(A)
    mur_pml = ufl.det(J) * A * mur * ufl.transpose(A)
    murinv_pml = ufl.inv(mur_pml)
    return (epsr_pml, murinv_pml)

x, y = ufl.SpatialCoordinate(mesh)
r = ufl.sqrt(x**2 + y**2)
x_stretched = pml_stretch(x, r, k0, x_dom=R_dom, x_pml=R_pml)
y_stretched = pml_stretch(y, r, k0, x_dom=R_dom, x_pml=R_pml)
x_pml = ufl.conditional(ufl.ge(abs(r), R_dom), x_stretched, x)
y_pml = ufl.conditional(ufl.ge(abs(r), R_dom), y_stretched, y)
pml_coords = ufl.as_vector((x_pml, y_pml))
epsr_pml, murinv_pml = pml_epsr_murinv(pml_coords, r)

def curl3d(a):
    curl_x = a[2].dx(1) 
    curl_y = - a[2].dx(0)
    curl_z = a[1].dx(0) - a[0].dx(1)
    return ufl.as_vector((curl_x, curl_y, curl_z))

def cross3dn(u, n):
    """Normal vector has only xy components."""
    w_x = - u[2]*n[1]
    w_y = u[2]*n[0]
    w_z = u[0]*n[1] - u[1]*n[0]
    return ufl.as_vector((w_x, w_y, w_z))

# Excitation and boundary conditions
def Eport(x):
    """Compute the normalized electric field distribution in all ports."""
    Ep = np.zeros((3, x.shape[1]), dtype=complex)
    for p in range(N_antennas):
        center = pos_antennas[p]
        phi = -phi_antennas[p] # Note rotation is by -phi here
        Rmat = np.array([[np.cos(phi), -np.sin(phi), 0],
                         [np.sin(phi), np.cos(phi), 0],
                         [0, 0, 1]])
        y = np.transpose(x.T - center)
        loc_x = np.dot(Rmat, y)
        Ep_tmp = np.vstack((0*loc_x[0], 0*loc_x[0], np.cos(kc*loc_x[0])))/np.sqrt(antenna_width/2)
        Ep_tmp[:,np.sqrt(loc_x[0]**2 + loc_x[1]**2 + loc_x[2]**2) > antenna_width] = 0
        Ep = Ep + Ep_tmp
    return Ep
Ep = dolfinx.fem.Function(Vspace)
Ep.sub(0).interpolate(lambda x: Eport(x)[0:2])
Ep.sub(1).interpolate(lambda x: Eport(x)[2])
pec_dofs = dolfinx.fem.locate_dofs_topological(Vspace, entity_dim=fdim, entities=boundaries.find(pec_surface_marker))
Ezero = dolfinx.fem.Function(Vspace)
Ezero.x.array[:] = 0.0
bc_pec = dolfinx.fem.dirichletbc(Ezero, pec_dofs)

# Set up simulation
E = ufl.TrialFunction(Vspace)
v = ufl.TestFunction(Vspace)
curl_E = curl3d(E)
curl_v = curl3d(v)
nvec = ufl.FacetNormal(mesh)
Zrel = dolfinx.fem.Constant(mesh, 1j)
k00 = dolfinx.fem.Constant(mesh, 1j)
a = [dolfinx.fem.Constant(mesh, 1.0 + 0j) for n in range(N_antennas)]
F_antennas_str = ''
for n in range(N_antennas):
    F_antennas_str += f"""+ 1j*k00/Zrel*ufl.inner(cross3dn(E, nvec), cross3dn(v, nvec))*ds_antennas[{n}] - 1j*k00/Zrel*2*a[{n}]*ufl.sqrt(Zrel*eta0)*ufl.inner(Ep, v)*ds_antennas[{n}]"""
F = ufl.inner(1/mur*curl_E, curl_v)*dx_dom \
    - ufl.inner(k00**2*epsr*E, v)*dx_dom \
    + ufl.inner(murinv_pml*curl_E, curl_v)*dx_pml \
    - ufl.inner(k00**2*epsr_pml*E, v)*dx_pml + eval(F_antennas_str)
bcs = [bc_pec]
lhs, rhs = ufl.lhs(F), ufl.rhs(F)
petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(
    lhs, rhs, bcs=bcs, petsc_options=petsc_options
)

def ComputeFields():
    S = np.zeros((Nf, N_antennas, N_antennas), dtype=complex)
    solutions = []
    for nf in range(Nf):
        print(f'Frequency {nf+1} / {Nf}')
        k00.value = 2*np.pi*fvec[nf]/c0
        Zrel.value = k00.value/np.sqrt(k00.value**2 - kc**2)
        sols = []
        for n in range(N_antennas):
            for m in range(N_antennas):
                a[m].value = 0.0
            a[n].value = 1.0
            E_h = problem.solve()
            for m in range(N_antennas):
                factor = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(2*ufl.sqrt(Zrel*eta0)*ufl.inner(Ep, Ep)*ds_antennas[m]))
                b = dolfinx.fem.assemble.assemble_scalar(dolfinx.fem.form(ufl.inner(cross3dn(E_h, nvec), cross3dn(Ep, nvec))*ds_antennas[m] + Zrel/(1j*k0)*ufl.inner(curl3d(E_h), cross3dn(Ep, nvec))*ds_antennas[m]))/factor
                S[nf,m,n] = b
            sols.append(E_h.copy())
        solutions.append(sols)
    return (S, solutions)

print('Computing REF solutions')
epsr.x.array[:] = epsr_array_ref
S_ref, solutions_ref = ComputeFields()
print('Computing DUT solutions')
epsr.x.array[:] = epsr_array_dut
S_dut, solutions_dut = ComputeFields()
delta_epsr = dolfinx.fem.Function(Wspace)

print('Computing optimization vectors')
Nepsr = len(delta_epsr.x.array[:])
A0 = np.zeros((Nf*N_antennas*N_antennas, Nepsr), dtype=complex)
A1 = np.zeros((Nf*N_antennas*N_antennas, Nepsr), dtype=complex)
A2 = np.zeros((Nf*N_antennas*N_antennas, Nepsr), dtype=complex)
A3 = np.zeros((Nf*N_antennas*N_antennas, Nepsr), dtype=complex)
b0 = np.zeros(Nf*N_antennas*N_antennas, dtype=complex)
b1 = np.zeros(Nf*N_antennas*N_antennas, dtype=complex)
b2 = np.zeros(Nf*N_antennas*N_antennas, dtype=complex)
b3 = np.zeros(Nf*N_antennas*N_antennas, dtype=complex)
# Create function spaces for temporary interpolation
q = dolfinx.fem.Function(Wspace)
bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
cell_volumes = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.conj(ufl.TestFunction(Wspace))*ufl.dx)).array
def q_func(x, Em, En, k0, conjugate=False):
    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x.T)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, x.T)
    for i, point in enumerate(x.T):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
    Em_vals_xy = Em.sub(0).eval(x.T, cells)
    Em_vals_z = Em.sub(1).eval(x.T, cells)
    En_vals_xy = En.sub(0).eval(x.T, cells)
    En_vals_z = En.sub(1).eval(x.T, cells)
    if conjugate:
        En_vals_xy = np.conjugate(En_vals_xy)
        En_vals_z = np.conjugate(En_vals_z)
    values = -1j*k0/eta0/2*(Em_vals_xy[:,0]*En_vals_xy[:,0] + Em_vals_xy[:,1]*En_vals_xy[:,1] + Em_vals_z[:,0]*En_vals_z[:,0])*cell_volumes
    return(values)
for nf in range(Nf):
    print(f'Frequency {nf+1} / {Nf}')
    k0 = 2*np.pi*fvec[nf]/c0
    for m in range(N_antennas):
        Em_ref = solutions_ref[nf][m]
        for n in range(N_antennas):
            En_dut = solutions_dut[nf][n]
            En_ref = solutions_ref[nf][n]
            # Case 0: Eref*Edut
            b0[nf*N_antennas*N_antennas + m*N_antennas + n] = S_dut[nf, m, n] - S_ref[nf, n, m]
            q.interpolate(functools.partial(q_func, Em=Em_ref, En=En_dut, k0=k0, conjugate=False))
            A0[nf*N_antennas*N_antennas + m*N_antennas + n,:] = q.x.array[:]
            # Case 1: Eref*Eref
            b1[nf*N_antennas*N_antennas + m*N_antennas + n] = S_dut[nf, m, n] - S_ref[nf, n, m]
            q.interpolate(functools.partial(q_func, Em=Em_ref, En=En_ref, k0=k0, conjugate=False))
            A1[nf*N_antennas*N_antennas + m*N_antennas + n,:] = q.x.array[:]
            # Case 2: Eref*conj(Edut)
            b2[nf*N_antennas*N_antennas + m*N_antennas + n] = np.conjugate(S_dut[nf, m, n]) - S_ref[nf, n, m]
            q.interpolate(functools.partial(q_func, Em=Em_ref, En=En_dut, k0=k0, conjugate=True))
            A2[nf*N_antennas*N_antennas + m*N_antennas + n,:] = q.x.array[:]
            # Case 3: Eref*conj(Eref)
            b3[nf*N_antennas*N_antennas + m*N_antennas + n] = np.conj(S_dut[nf, m, n]) - S_ref[nf, n, m]
            q.interpolate(functools.partial(q_func, Em=Em_ref, En=En_ref, k0=k0, conjugate=True))
            A3[nf*N_antennas*N_antennas + m*N_antennas + n,:] = q.x.array[:]

# Save values for further postprocessing
np.savez('output.npz', A0=A0, A1=A1, A2=A2, A3=A3, b0=b0, b1=b1, b2=b2, b3=b3, fvec=fvec, S_ref=S_ref, S_dut=S_dut, epsr_mat=epsr_mat, epsr_defect=epsr_defect, epsr_array_ref=epsr_array_ref, epsr_array_dut=epsr_array_dut)

#exit()

E_h = solutions_dut[-1][0]

# Plot the field in the domain
PlotSpace = dolfinx.fem.functionspace(mesh, ('CG', 1))
E_plot = dolfinx.fem.Function(PlotSpace)
def CreateGrid(E):
    E_plot.interpolate(dolfinx.fem.Expression(E, PlotSpace.element.interpolation_points()))
    E_plot_array = E_plot.x.array
    cells, cell_types, x = dolfinx.plot.vtk_mesh(mesh, tdim)
    E_grid = pv.UnstructuredGrid(cells, cell_types, x)
    E_grid["plotfunc"] = np.real(E_plot_array)
    return (E_grid.copy(), E_plot_array.copy())
        
def sym_clim(E):
    E_max = np.max(np.abs(E))
    clim = [-E_max, E_max]
    return clim

Ex_grid, Ex_array = CreateGrid(E_h[0])
Ey_grid, Ey_array = CreateGrid(E_h[1])
Ez_grid, Ez_array = CreateGrid(E_h[2])
clim_Ex = sym_clim(Ex_array)
clim_Ey = sym_clim(Ey_array)
clim_Ez = sym_clim(Ez_array)

# Indicate material region
V = dolfinx.fem.functionspace(mesh, ('DG', 0))
u = dolfinx.fem.Function(V)
mat_cells = subdomains.find(mat_marker)
mat_dofs = dolfinx.fem.locate_dofs_topological(V, entity_dim=tdim, entities=mat_cells)
u.x.array[:] = 0
u.x.array[mat_dofs] = 0.25 # Used as opacity value later
cells, cell_types, x = dolfinx.plot.vtk_mesh(mesh, tdim)
mat_grid = pv.UnstructuredGrid(cells, cell_types, x)
mat_grid["u"] = np.real(u.x.array)

plotter = pvqt.BackgroundPlotter(shape=(1,3), window_size=(3840,2160), auto_update=True)
def AddGrid(E_grid, title='', clim=None):
    plotter.add_mesh(E_grid, show_edges=False, show_scalar_bar=True, scalar_bar_args={'title': title, 'title_font_size': 12, 'label_font_size': 12}, clim=clim, cmap='bwr')
    plotter.add_mesh(mat_grid, show_edges=False, scalars='u', opacity='u', cmap='binary', show_scalar_bar=False)
    plotter.view_xy()
    plotter.add_text(title, font_size=18)
    plotter.add_axes()

plotter.subplot(0, 0)
AddGrid(Ex_grid, title='E_x', clim=clim_Ex)
plotter.subplot(0, 1)
AddGrid(Ey_grid, title='E_y', clim=clim_Ey)
plotter.subplot(0, 2)
AddGrid(Ez_grid, title='E_z', clim=clim_Ez)
Nphase = 120
phasevec = np.linspace(0, 2*np.pi, Nphase)
plotter.open_movie('tmp.mp4')
for phase in phasevec:
    Ex_grid["plotfunc"] = np.real(Ex_array*np.exp(1j*phase))
    Ey_grid["plotfunc"] = np.real(Ey_array*np.exp(1j*phase))
    Ez_grid["plotfunc"] = np.real(Ez_array*np.exp(1j*phase))
    plotter.app.processEvents()
    plotter.write_frame()
plotter.close()
