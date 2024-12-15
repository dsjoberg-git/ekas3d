# ekas3d
Files for simulating a scattering geometry and locate defects in the EKAS3D project.

## Installation

The installation instructions below are based on the instructions for installing [fe2ms](https://github.com/nwingren/fe2ms), a FE-BI hybrid code developed by Niklas Wingren. 

The code is primarily based on FEniCSx which is available on macOS and Linux. However, installation of this package has only been tested on Ubuntu and the installation instructions are written for this. For Windows users, Linux can be run easily using Windows Subsystem for Linux (WSL, make sure you have WSL2, you may need to enable Virtual Machine Feature). Installation instructions and more information can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install).

Installation using mamba (similar to conda) is recommended. The instructions are as follows.

### Install mamba

Please follow [these](https://github.com/conda-forge/miniforge#mambaforge) instructions to install mamba. Following this, it is recommended that you create a new environment as follows ("ekas3d" can be changed to your preferred environment name).

```bash
mamba create --name ekas3d
mamba activate ekas3d
```

### Install FEniCSx

Having activated the ekas3d environment, the following will install fenicsx there.

```bash
mamba install fenics-dolfinx mpich petsc=*=complex*
```

### Install other Python packages

This will install other Python packages into the ekas3d environment. ```imageio``` seems to need to be installed through pip instead of mamba. 

```bash
mamba install scipy matplotlib python-gmsh pyvista pyvistaqt spgl1 h5py petsc=*=complex*
pip install imageio[ffmpeg]
```

### Install some optional packages

Paraview is a visualization program that can be convenient, but not necessary, to use. Something seems broken in the mamba installation, but it can be installed directly in the system, rather than in the environment. Also vlc (for viewing videos) seems to be easier to install in the system.

```bash
sudo apt update
sudo apt install paraview
sudo apt install vlc
```


## Files

- ```scatt2d.py``` is the main simulation code. This creates the geometry and mesh, and computes the scattering problem for the reference case (no defects) and the DUT case (with defects). It also computes the optimization vectors. This can be run in parallel with ```mpirun -n 4 python scatt2d.py```.
- ```postprocess.py``` runs the optimization to identify the defects. This has not been evaluated in parallel yet.

## Author

Daniel Sjöberg, Lund University. [daniel.sjoberg@eit.lth.se](mailto:daniel.sjoberg@eit.lth.se)
