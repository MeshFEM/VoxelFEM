import numpy as np
import enum
import pyVoxelFEM # must call before import benchmark
import symmetry_utils
from benchmark import benchmarkit, benchmarkit_customname
from CoarseningLevelBenchmark import optMGLevel

class LayerByLayerObjective:
    """
    Class for computing and differentiating the total compliance under
    self-weight of all the active intermediate shapes (full domain (= active shape + void region) preserved)
    generated when 3D printing a given voxel object.
    """
    BUILD_DIRECTION = 1 # y axis as the building direction for both 2D/3D
    class _InterpolationLaw(enum.Enum):
        SIMP = pyVoxelFEM.InterpolationLaw.SIMP
        RAMP = pyVoxelFEM.InterpolationLaw.RAMP
    intpLaw = _InterpolationLaw.RAMP
    pcgVars = {'maxIter': 50, 'tol': 1e-5, 'mgIterations': 1, 'mgSmoothingIterations': 1, 'it_callback': None, 'fullMultigrid': False}
    default_ramp_q  = 3
    zeroInit = True # for full object, initialize from previous optimization step
    cppEvaluator = None
    cutCell = True # compare constant zeroOutDensity and maskedHeight(cutCell)

    def __get_init_method(self): return self.__init_method
    def __set_init_method(self, method): self.__init_method = method
    init_method = property(__get_init_method, __set_init_method)
    def newSolver(self, tps):
        mg_levels = self.mg_levels
        if mg_levels is None: mg_levels = optMGLevel(tps.NbElementsPerDimension, len(tps.NbElementsPerDimension))
        return tps.multigridSolver(mg_levels)
    def mgSolve(self, u_guess, f):
        return self.layerByLayerMGSolver.preconditionedConjugateGradient(u_guess, f, **self.pcgVars)
    @benchmarkit
    def setFabricationMaskHeightByLayer(self, layer):
        self.layerByLayerMGSolver.setFabricationMaskHeightByLayer(layer)
    # Modify some properties
    def setCGTol(self, tol: float): self.pcgVars['tol'] = tol # impact future simulations
    def setQ(self, q: float):
        E_min, E_0 = self.layerByLayerSimulator.E_min, self.layerByLayerSimulator.E_0
        if q <= 0 or q >= (E_0 - E_min) / E_min: return
        self.layerByLayerSimulator.q = q
    def setIntptLaw(self, law: _InterpolationLaw): # impact tps
        if not isinstance(law, self._InterpolationLaw): return
        self.intpLaw = law
        self.layerByLayerSimulator.interpolationLaw = law.value

    def __init__(self, tps, mg_levels=None, init_method='N=3', downsampling_levels=0, symmetry_axes_mask=None, pcgVars={}):
        """
        Construct a LayerByLayerObjective for an existing simulator object.

        Parameters
        ----------
        tps : TensorProductSimulator
            The full object to be sliced and printed
        mg_levels : int or None
            Number of levels in the multigrid hierarchy build for each intermediate shape
            If `None`, we set this automatically based on each simulator's grid size.
        downsampling_levels : int
            Number of 2-1 downsamplings to apply when constructing the layer simulation objects.
            If `downsampling_levels = 0`, then the full input resolution will be used.
            If `downsampling_levels = l`, then a resolution 2**l times lower will be used
        method: str
            Methods to use for guess at each layer, eg, 'constant', 'first', 'N=2', etc.
        pcgVars : dict
            Parameters for MGPCG solver with default values {'tol': 1e-5, 'maxIter': 50, 'mgIterations': 1, 'mgSmoothingIterations': 1, 'it_callback'=None},
            any of which can be overwritten.
        symmetry_axes_mask : boolean array
            Whether to apply symmetry conditions across the max-X and max-Z boundaries.
            (`True` in position `d` means the conditions are applied to the max-d face.)
            When symmetry conditions are applied, we effectively simulate a larger
            reflected object (the compliance computed in the symmetry base cell is scaled
            by the number of copies it reflects into).
        """
        if (tps.E_min < 1e-5): print('WARNING: high contrast ratios suffer from poor multigrid convergence')
        self.DIM                    = len(tps.NbElementsPerDimension)
        self.highresSimulator       = tps # keep a class reference to the original high resolution tps
        self.mg_levels              = mg_levels
        if (downsampling_levels < 0): raise Exception('Number of downsampling levels must be non negative')
        self.downsampling_levels    = downsampling_levels
        symmetry_axes_mask          = symmetry_utils.validatedSymmetryAxes(self.DIM, symmetry_axes_mask)
        if symmetry_axes_mask[self.BUILD_DIRECTION]: raise Exception('Illegal symmetry_axes_mask: the build platform cannot be a symmetry plane')
        self.symmetry_axes_mask     = symmetry_axes_mask
        self.numVoxelLayers         = int(tps.NbElementsPerDimension[self.BUILD_DIRECTION]) // 2**downsampling_levels # number of layers that will be simulated
        self.pcgVars.update(pcgVars)
        if self.downsampling_levels > 0: # downsample to a lower resolution object (densities NOT set)
            self.layerByLayerSimulator = self.highresSimulator.downsample(self.downsampling_levels).getIntermediateFabricationShape(1, False, self.intpLaw.value)
        else:
            self.layerByLayerSimulator = self.highresSimulator.getIntermediateFabricationShape(1, False, self.intpLaw.value)
        self.layerByLayerSimulator.q = self.default_ramp_q
        if self.symmetry_axes_mask.sum() != 0: # Must happen before building the MG solver!
            self.layerByLayerSimulator.applySymmetryConditions(self.symmetry_axes_mask, [1] * self.DIM) # `1` means max face
        self.layerByLayerMGSolver = self.newSolver(self.layerByLayerSimulator)
        self.setDensities()
        self.init_method = init_method

    @benchmarkit
    def setDensities(self, densities=None): # Followed by one full simulation of layers
        # Reset simulator and MGPCG
        if densities is None: densities = self.highresSimulator.getDensities()
        self.totalCompliances           = 0
        self.totalComplianceGradients   = 0
        # pass current or specified densities to downsampled layer-by-layer obj
        self.highresSimulator.downsampleDensityFieldTo(densities, self.layerByLayerSimulator)

    @benchmarkit
    def simulate(self, increment=1, verbose = False, lblCallback=None):
        tps, mg = self.layerByLayerSimulator, self.layerByLayerMGSolver
        if self.cppEvaluator is None:
            self.cppEvaluator = pyVoxelFEM.LayerByLayerEvaluator(tps)
        self.cppEvaluator.selectInitMethod(self.init_method)
        self.cppEvaluator.run(mg, self.zeroInit, increment, **self.pcgVars, verbose=verbose, lblCallback=lblCallback)
        self.totalCompliances = self.cppEvaluator.objective()
        self.totalComplianceGradients = self.cppEvaluator.gradient()

    def isDownsampling(self): return self.downsampling_levels > 0

    # compute elastic energy: 1/2 u^T K u - u^T g
    def elasticEnergy(self, tps, u, f): return u.ravel().dot(tps.applyK(u).ravel()) / 2 - u.ravel().dot(f.ravel())

    def numReflectedCopies(self): return symmetry_utils.numReflectedCopies(self.symmetry_axes_mask)
    def compliancePerLayer(self, u, f):
        return 0.5 * u.ravel().dot(f.ravel())
    # Derivative of 1/2 f^l . u^l
    def gradientPerLayer(self, u):
        return self.layerByLayerSimulator.complianceGradient(u)

    @benchmarkit_customname('LayerByLayerObjective setVars')
    def setVars(self, x, skipLayers=1, verbose=False, lblCallback=None): # simulate the full object only
        self.setDensities(x)
        self.simulate(increment=skipLayers, verbose=verbose, lblCallback=lblCallback)
    def energy(self):
        return self.totalCompliances * self.numReflectedCopies()
    def gradient(self):
        result = self.totalComplianceGradients
        if (self.isDownsampling()):
            result = self.highresSimulator.upsampleDensityGradientFrom(self.layerByLayerSimulator, result) # upsample
        return result * self.numReflectedCopies()

