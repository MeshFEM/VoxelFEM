import numpy as np

def interpolate(densities, newGridDim, method='linear'):
    """
    Upsample a density field `densities` to new grid dimensions `newGridDim`.
    """
    import numbers, scipy.interpolate

    if isinstance(newGridDim, numbers.Number):
        newGridDim = [newGridDim] * densities.ndim

    points = [np.linspace(0, 1, s) for s in densities.shape]
    
    X, Y, Z = np.meshgrid(*[np.linspace(0, 1, s) for s in newGridDim], indexing='ij')
    xi = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    return scipy.interpolate.interpn(points, densities, xi, method=method).reshape(*newGridDim)

def binarize(values, threshold = 0.5):
    """
    Convert a grayscale design to black and white using a specified threshold
    """
    return values >= threshold

def extrude(values, voxelHeight):
    """
    Extrude a 2D density field along the "y" axis making `voxelHeight` copies.
    """
    return np.repeat(values[:, np.newaxis, :], voxelHeight, axis=1)

def contour(design, dx):
    """
    Extract a triangle mesh representing the design's boundaries (the transition
    from below density 0.5 to above.

    `dx` specifies the edge length of each grid cell (all grid cells are assumed to
    be identical cubes).
    """
    import igl

    # Support both single-scalar (cube) and per-dimension `dx`.
    dim = len(design.shape)
    if hasattr(dx, "__len__"):
        if len(dx) != dim: raise Exception('dx and grid shape mismatch')
    else: dx = [dx] * dim

    # Pad on all sides with a density of zero (distance of 1) to ensure the
    # mesh is watertight even with boundary densities filled in. (libigl
    # does not mesh the intersection of the object with the marching cubes grid
    # boundary).
    signed_distance = np.pad(0.5 - design, [[1, 1]] * 3, constant_values=0.5)

    # Note that libigl's marching cubes effectively expects a `Fortran` storage ordering
    # for the gridpoint array:
    #        [x, y, z] ==> x + nx * (y + ny * z)
    # The signed distance values are sampled at the **voxel centers** of the padded grid, which are:
    #   |-------|------|-------|-------|....
    #     -dx/2   dx/2   3dx/2   5dx/2
    mcGridPoints = np.column_stack([C.ravel(order='F') for C in np.meshgrid(*([np.linspace(-dx_i / 2, dx_i * (n - 1.5), n) for dx_i, n in zip(dx, signed_distance.shape)]), indexing='ij')])
    return igl.marching_cubes(signed_distance.ravel(order='F'), mcGridPoints, *(signed_distance.shape))
