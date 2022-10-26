VoxelFEM
=========

<img src='http://julianpanetta.com/publication/layer_by_layer/teaser_hu1f7de7a219193e71c3c4bc0e101b48bb_204536_720x0_resize_lanczos_3.png' width='100%'/>

This is the codebase for our SCF 2022 paper,
[Efficient Layer-by-Layer Simulation for Topology Optimization](http://julianpanetta.com/publication/layer_by_layer/).
It includes a high-performance topology optimization code for linear elastic compliance minimization
(using a multigrid preconditioned CG solver) as well as our algorithms for accelerating layer-by-layer simulation.

The code is written primarily in C++, but it is meant to be used through the Python bindings.

# Acknowledgements
We thank Michele Vidulis, Shad Durussel, and Vincent Pollet for their
contributions to this multigrid topology optimization framework.
