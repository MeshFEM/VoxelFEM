'''
In 2D problems, coarsenest grid of dimension 64 x 32 performs very well.
In 3D problems, coarsenest grid of dimension 16 x 8 x 8 performs very well.
Let "opt_dim" = 32 for 2D or 8 for 2D.
Assume the aspect ratio is 2:1 or 2:1:1.
'''
import pyVoxelFEM
import MeshFEM, mesh
import numpy as np
import matplotlib.pyplot as plt
import benchmark, time
import argparse, os

def mgpcg(mg, mg_ite=1, sm_ite=1, full_mg=True):
        residuals = []
        time_elapsed = []
        def it_callback(it, u_curr, residual): # iteration, approximated sol at iteration, residual at each iteration
            time_elapsed.append(time.time() - start_time)
            residuals.append(np.linalg.norm(residual))
            #print(it, time.time() - start_time, np.linalg.norm(residual))
        start_time = time.time()
        u_mg = mg.preconditionedConjugateGradient(u_guess, f,
                                                maxIter=100,             # Maximum number of iterations to run (if solver hasn't converged yet)
                                                tol=1e-10,                # Convergence tolerance on the norm of the residual
                                                it_callback=it_callback,
                                                mgIterations=mg_ite,          # Number of multigrid iterations to run for the preconditioner
                                                mgSmoothingIterations=sm_ite, # Number of relaxation steps to run at each level before restriction/after interpolation
                                                fullMultigrid=full_mg      # Whether to use full multigrid or a plain V-cycle
                                                )
        time_elapsed = [t - time_elapsed[0] for t in time_elapsed]
        return time_elapsed, residuals

def possibleMaxMGLevel(dims):
    """
    Determine possible number of coarsening levels for the multigrid hierarchy.
    """
    largestPow2Divisor = lambda n: (n & (~(n - 1)))
    maxLevel = min([int(np.log2(largestPow2Divisor(int(d)))) for d in dims])
    return maxLevel

def optMGLevel(dims, DIM):
    '''
    Determine optimal number l of coarsening levels.
    Assume the finest dimension of shorter edge = 2^n M for some M odd and n>=1, then the max number of coarsening levels
    allowed is n. We hope to find an optimal m such that the coarsenest grid dimension is close to "opt_dim".
    Then the strategy is as follows:
    if 2^n M <= opt_dim: 1 level.
    else if M < opt_dim: m levels such that |2^m M - opt_size| is minimized.
    else: n levels.
    Note that 2^l M <= opt_dim <= 2^{l+1} M <=> ... <=> l <= log2(opt_dim/M) <= l+1

    Parameters
    ----------
    dims: number of elements per edge
    DIM: 2 or 3
    '''
    opt_dim = 16 if DIM == 2 else 8
    if min(dims) <= opt_dim:
        if min(dims) == 1: return 0
        return 1
    n = possibleMaxMGLevel(dims)
    M = min(dims)/pow(2, n)
    if M < opt_dim:
        m = np.log2(opt_dim/M)
        return round(n - m)
    return n

if __name__ == '__main__':
    parser = argparse.ArgumentParser('User defined grid dimensions')
    parser.add_argument ("dimension",   default=2,    choices=[2, 3],   type=int,  help="Dimension of the problem: 2 or 3")
    parser.add_argument("resolution",   default="192,96",               type=str,  help="Grid dimensions: ie, 192,96 or 192,96,96")
    options = parser.parse_args()
    # 2D or 3D? what dimensions?
    DIM = options.dimension
    gridDimensions = [eval(i) for i in options.resolution.split(',')]
    # Material and "boundary conditions" (can be imposed also inside the domain) are read from file
    BC_PATH = os.path.join('..', 'examples', 'bcs', '3D' if DIM ==3 else '', 'cantilever_flexion_E.bc')
    # Domain and mesh
    FEMDegree = [1] * DIM # Basis function degree
    domainCorners = [[0, 0], [2, 1]] if DIM == 2 else [[0, 0, 0], [2, 1, 1]]

    # Initialize simulator, define material and apply external constraints
    tps = pyVoxelFEM.TensorProductSimulator(FEMDegree, domainCorners, gridDimensions)
    tps.readMaterial(os.path.join('..', 'examples', 'materials', 'B9Creator.material'))
    tps.applyDisplacementsAndLoadsFromFile(BC_PATH)
    tps.E_min = 1e-5

    # Set densities
    uniformDensity = 0.5
    tps.setUniformDensities(uniformDensity)
    u_guess = np.zeros((tps.numNodes(), DIM))
    # Construct right-hand side vector corresponding to the boundary conditions
    f = tps.buildLoadVector()
    dims = tps.NbElementsPerDimension
    nlevels = range(optMGLevel(dims, DIM), possibleMaxMGLevel(dims)+1) # collecy all the possible number of levels

    results = []
    for level in nlevels:
        mg = tps.multigridSolver(level) # Build a multigrid solver for the `tps` object
        results.append(mgpcg(mg))

    title = ("coarsen_level_{0[0]}x{0[1]}".format(gridDimensions) if DIM == 2
                else "coarsen_level_{0[0]}x{0[1]}x{0[2]}".format(gridDimensions))
    plt.title(title)
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    for result, level in zip(results, nlevels):
        plt.scatter(result[0], result[1], label=level, s=10)
    plt.legend(title='Levels')
    plt.savefig(os.path.join('benchmark_out', title))
