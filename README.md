VoxelFEM
=========

<img src='http://julianpanetta.com/publication/layer_by_layer/teaser_hu1f7de7a219193e71c3c4bc0e101b48bb_204536_720x0_resize_lanczos_3.png' width='100%'/>

This is the codebase for our SCF 2022 paper,
[Efficient Layer-by-Layer Simulation for Topology Optimization](http://julianpanetta.com/publication/layer_by_layer/).
It includes a high-performance topology optimization code for linear elastic compliance minimization
(using a multigrid preconditioned CG solver) as well as our algorithms for accelerating layer-by-layer simulation.

The code is written primarily in C++, but it is meant to be used through the Python bindings.

# Getting Started
## C++ Code Dependencies

The C++ code relies on `Boost` and `CHOLMOD/UMFPACK`, which must be installed
separately.

The code also relies on [MeshFEM](https://github.com/MeshFEM/MeshFEM), which is included as a submodule.

### macOS
You can install all the mandatory dependencies on macOS with [MacPorts](https://www.macports.org).

```bash
# Build/version control tools, C++ code dependencies
sudo port install cmake boost ninja
sudo port install SuiteSparse +accelerate
# Dependencies for jupyterlab/notebooks
sudo port install python39
# Install nodejs/npm using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install 17 && nvm use 17
```

### Ubuntu 20.04
A few more packages need to be installed on a fresh Ubuntu 20.04 install:
```bash
# Build/version control tools
sudo apt install git cmake ninja-build
# Dependencies for C++ code
sudo apt install libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libsuitesparse-dev
# Dependencies (pybind11, jupyterlab/notebooks)
sudo apt install python3-pip npm
sudo npm install npm@latest -g
```

## Obtaining and Building

Clone this repository *recursively* so that its submodules are also downloaded:

```bash
git clone --recursive https://github.com/MeshFEM/VoxelFEM
```

Build the C++ code and its Python bindings using `cmake` and your favorite
build system. For example, with [`ninja`](https://ninja-build.org):

```bash
cd VoxelFEM
mkdir build && cd build
cmake .. -GNinja
ninja
```

## Running the Jupyter Notebooks
The preferred way to interact with VoxelFEM is in a Jupyter notebook,
using the Python bindings.
We recommend that you install the Python dependencies and JupyterLab itself in a
virtual environment (e.g., with [venv](https://docs.python.org/3/library/venv.html)).

```bash
pip3 install wheel # Needed if installing in a virtual environment
# Recent versions of jupyterlab and related packages cause problems:
#   JupyerLab 3.4 and later has a bug where the tab and status bar GUI
#                 remains visible after taking a viewer fullscreen
#   ipykernel > 5.5.5 clutters the notebook with stdout content
#   ipywidgets 8 and juptyerlab-widgets 3.0 break pythreejs
pip3 install jupyterlab==3.3.4 ipykernel==5.5.5 ipywidgets==7.7.2 jupyterlab-widgets==1.1.1
# If necessary, follow the instructions in the warnings to add the Python user
# bin directory (containing the 'jupyter' binary) to your PATH...

git clone https://github.com/jpanetta/pythreejs
cd pythreejs
pip3 install -e .
cd js
jupyter labextension install .

pip3 install matplotlib scipy
```

You may need to add the following to your shell startup script for the installation of `pythreejs`'s dependencies during `pip3 install -e .` to succeed:
```
export NODE_OPTIONS=--openssl-legacy-provider;
```

Launch JupyterLab from the root python directory:
```bash
cd python
jupyter lab
```

Now try opening and running an demo notebook, e.g.,
[`python/3DTopoptDemo.ipynb`](https://github.com/MeshFEM/VoxelFEM/blob/main/python/3DTopoptDemo.ipynb).

# Acknowledgements
We thank Michele Vidulis, Shad Durussel, and Vincent Pollet for their
contributions to this multigrid topology optimization framework.
