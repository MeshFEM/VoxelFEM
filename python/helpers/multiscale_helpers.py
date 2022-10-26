import numpy as np

def upscaleScalarField(gridDimensions_old, x_old):
    """
    Receive a density field and double the number elements 
    in each of the coordinate directions.
    """
    nelx = gridDimensions_old[0]
    nely = gridDimensions_old[1]
    x = np.zeros((4*x_old.shape[0]))
    for j in range(2*nelx):
        if j % 2 == 0:
            for i in range(2*nely):
                x[i + 2*nely*j] = x_old[int(i/2) + nely*int(j/2)]
        elif j % 2 == 1:
            x[2*nely*j:2*nely*(j+1)] = x[2*nely*(j-1):2*nely*j]
    return (2*nelx, 2*nely), x   


def downscaleScalarField(gridDimensions_old, x_old):
    """
    Receive a density field and halve the number elements 
    in each of the coordinate directions.
    """
    nelx = gridDimensions_old[0]
    nely = gridDimensions_old[1]
    new_nelx = int(nelx/2)
    new_nely = int(nely/2)
    x = np.zeros(new_nelx*new_nely)
    for j in range(new_nelx):
        for i in range(new_nely):
            x[i + new_nely*j] = x_old[2*i + nely*2*j]
    return (new_nelx, new_nely), x