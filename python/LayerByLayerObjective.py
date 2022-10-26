import numpy as np
import scipy as sp
import time
import enum
import pyVoxelFEM # must call before import benchmark
import symmetry_utils
import benchmark
from benchmark import benchmarkit, benchmarkit_customname
from matplotlib import pyplot as plt
from CoarseningLevelBenchmark import optMGLevel
from helpers.write_benchmark import write_benchmark_report

class LayerByLayerObjective:
    """
    Class for computing and differentiating the total compliance under
    self-weight of all the active intermediate shapes (full domain (= active shape + void region) preserved)
    generated when 3D printing a given voxel object.
    """
    BUILD_DIRECTION = 1 # y axis as the building direction for both 2D/3D
    class _MGType(enum.Enum): # Using MG or AMG
        MG = 0
        AMG = 1
    class _InterpolationLaw(enum.Enum):
        SIMP = pyVoxelFEM.InterpolationLaw.SIMP
        RAMP = pyVoxelFEM.InterpolationLaw.RAMP

    mgType  = _MGType.MG
    intpLaw = _InterpolationLaw.RAMP
    pcgVars = {'maxIter': 50, 'tol': 1e-5, 'mgIterations': 1, 'mgSmoothingIterations': 1, 'it_callback': None}
    monitorVars = {'numCGs': False, 'residualNorms': False, 'errorNorms': False, 'coeffs': False, 'densities': False, 'plot': False, 'visualization': False}
    default_ramp_q  = 3
    zeroInit = True # for full object, initialize from previous optimization step
    cppEvaluator = None
    cutCell = True # compare constant zeroOutDensity and maskedHeight(cutCell)

    def newSolver(self, tps):
        if self.mgType == self._MGType.MG:
            mg_levels = self.mg_levels
            if mg_levels is None: mg_levels = optMGLevel(tps.NbElementsPerDimension, len(tps.NbElementsPerDimension))
            return tps.multigridSolver(mg_levels)
        if self.mgType == self._MGType.AMG:
            return tps.AMGCLSolver()
    def mgSolve(self, u_guess, f):
        if self.mgType == self._MGType.AMG: return self.layerByLayerMGSolver.solve(u_guess, f, **self.pcgVars)
        if self.mgType == self._MGType.MG: return self.layerByLayerMGSolver.preconditionedConjugateGradient(u_guess, f, **self.pcgVars)
    @benchmarkit
    def setFabricationMaskHeightByLayer(self, layer):
        if self.mgType == self._MGType.MG: self.layerByLayerMGSolver.setFabricationMaskHeightByLayer(layer)
        if self.mgType == self._MGType.AMG: self.layerByLayerSimulator.setFabricationMaskHeightByLayer(layer)
    # Modify some properties
    def setCGTol(self, tol: float): self.pcgVars['tol'] = tol # impact future simulations
    def setQ(self, q: float):
        E_min, E_0 = self.layerByLayerSimulator.E_min, self.layerByLayerSimulator.E_0
        if q <= 0 or q >= (E_0 - E_min) / E_min: return
        self.layerByLayerSimulator.q = q
    def setMGType(self, type: _MGType): # only impact MG solvers
        if not isinstance(type, self._MGType): return
        self.mgType = type
        self.layerByLayerMGSolver = self.newSolver(self.layerByLayerSimulator)
    def setIntptLaw(self, law: _InterpolationLaw): # impact tps
        if not isinstance(law, self._InterpolationLaw): return
        self.intpLaw = law
        self.layerByLayerSimulator.interpolationLaw = law.value

    def __init__(self, tps, mg_levels=None, init_method='N=3', enrich_mode=None, downsampling_levels=0, symmetry_axes_mask=None, fd_validation_support=False, pcgVars={}, monitorVars={}):
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
        fd_validation_support : bool
            Whether to store an additional copy of the density variables needed to support
            finite difference validation of the density coarsening/gradient upscaling.
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
        self.__fd_support           = fd_validation_support
        self.numVoxelLayers         = int(tps.NbElementsPerDimension[self.BUILD_DIRECTION]) // 2**downsampling_levels # number of layers that will be simulated
        self.pcgVars.update(pcgVars)
        self.monitorVars.update(monitorVars)
        if self.downsampling_levels > 0: # downsample to a lower resolution object (densities NOT set)
            self.layerByLayerSimulator = self.highresSimulator.downsample(self.downsampling_levels).getIntermediateFabricationShape(1, False, self.intpLaw.value)
        else:
            self.layerByLayerSimulator = self.highresSimulator.getIntermediateFabricationShape(1, False, self.intpLaw.value)
        self.layerByLayerSimulator.q = self.default_ramp_q
        if self.symmetry_axes_mask.sum() != 0: # Must happen before building the MG solver!
            self.layerByLayerSimulator.applySymmetryConditions(self.symmetry_axes_mask, [1] * self.DIM) # `1` means max face
        self.layerByLayerMGSolver = self.newSolver(self.layerByLayerSimulator)

        self.setDensities()
        self.initializer = self.InitializationGenerator(self, self.layerByLayerSimulator, init_method, enrich_mode) # Used to generate initialization

    @benchmarkit
    def setDensities(self, densities=None): # Followed by one full simulation of layers
        ''' Reset simulator and MGPCG'''
        if densities is None: densities = self.highresSimulator.getDensities()
        self.totalCompliances           = 0
        self.totalComplianceGradients   = 0

        # Mask height will be set when simulating each layer, NO need to set it here

        if self.__fd_support: self.vars = densities.copy()
        # pass current or specified densities to downsampled layer-by-layer obj
        self.highresSimulator.downsampleDensityFieldTo(densities, self.layerByLayerSimulator)
        if True in self.monitorVars.values():
            self.convergenceMonitor = self.ConvergenceMonitor(self)
            self.pcgVars['it_callback'] = self.convergenceMonitor.record
        else: self.convergenceMonitor = None
    @benchmarkit
    def simulate(self, start=None, end=None, increment=1, useCPP = True, verbose = False, lblCallback=None):
        tps, mg = self.layerByLayerSimulator, self.layerByLayerMGSolver
        if useCPP:
            if self.cppEvaluator is None:
                self.cppEvaluator = pyVoxelFEM.LayerByLayerEvaluator(tps)
            self.cppEvaluator.selectInitMethod(self.initializer.init_method)
            self.cppEvaluator.run(mg, self.zeroInit, increment, **self.pcgVars, verbose=verbose, lblCallback=lblCallback)
            self.totalCompliances = self.cppEvaluator.objective()
            self.totalComplianceGradients = self.cppEvaluator.gradient()
            return

        start = self.numVoxelLayers if start is None else start
        end = 0 if end is None else end
        for layer in range(start, end, -increment): # from full object to object with one solid layer
            if self.cutCell: self.setFabricationMaskHeightByLayer(layer)
            else: tps.setDensities(self.densitiesWithZeroTopLayers(tps.getDensities(), layer+1))
            benchmark.start_timer_section("Build load")
            f = tps.buildLoadVector() # build self weight
            benchmark.stop_timer_section("Build load")
            benchmark.start_timer_section("Construct initial guess")
            self.initializer.processing_layer = layer
            u_guess = self.initializer.nextGuess()
            benchmark.stop_timer_section("Construct initial guess")

            if self.convergenceMonitor is not None:
                densities = self.densitiesWithZeroTopLayers(tps.getDensities(), layer, tps) if self.monitorVars['densities'] else None
                self.convergenceMonitor.reset(layer, u_guess, f, densities)
            u = self.mgSolve(u_guess, f)
            self.totalCompliances += self.compliancePerLayer(u, f) # Sum up compliance
            self.totalComplianceGradients += self.gradientPerLayer(u) # Sum up compliance gradients

            benchmark.start_timer_section("Rest of simulating")
            self.initializer.addToHistory(u)
            if layer == self.numVoxelLayers and not self.zeroInit: self.initializer.prev_opt = u
            if self.convergenceMonitor is not None:
                self.convergenceMonitor.postprocess(u, u_guess, f)
            benchmark.stop_timer_section("Rest of simulating")
        self.totalCompliances /= self.initializer.simulations_done
        self.totalComplianceGradients /= self.initializer.simulations_done

    def isDownsampling(self): return self.downsampling_levels > 0

    def toNodeMultiIndex(self, arr, tps=None):
        if tps is None: tps = self.layerByLayerSimulator
        return arr.reshape(tuple(tps.NbNodesPerDimension) + (self.DIM,)) # Nx x Ny (x Nz) x DIM
    def toNodeFlatIndex(self, arr, tps=None):
        if tps is None: tps = self.layerByLayerSimulator
        return arr.reshape((tps.numNodes(), self.DIM)) # numNodes x DIM
    def toEleMultiIndex(self, arr, tps=None):
        if tps is None: tps = self.layerByLayerSimulator
        return arr.reshape(tuple(tps.NbElementsPerDimension)) # Nx x Ny (x Nz)
    def toEleFlatIndex(self, arr, tps=None):
        if tps is None: tps = self.layerByLayerSimulator
        return arr.reshape(tps.numElements()) # numElements
    def densitiesWithZeroTopLayers(self, densities, l, tps=None):
        copy = densities.copy()
        self.zeroOutTopLayersForDensity(copy, l, tps)
        return copy
    def zeroOutTopLayersForDensity(self, densities, l, tps=None):
        ''' Zero out densities at layers l, l+1, l+2, ... Used Masked height instead. This is only used for simulating densities.
        '''
        densities = self.toEleMultiIndex(densities, tps)
        densities[:, l:] = 0  # in 3D equivalent to densities[:, l:, :]
    def zeroOutTopLayersForDisplacement(self, u, l, tps=None):
        u = self.toNodeMultiIndex(u, tps)
        u[:, l:] = 0

    class InitializationGenerator: # Generate the initual guess
        def __init__(self, obj, tps, init_method, enrich_mode=None):
            self.obj              = obj
            self.tps              = tps
            self.init_method      = init_method
            self.enrich_mode      = enrich_mode
            self.minimize         = 'energy'
            self.simulations_done = 0 # can be reset in simulation
            self.processing_layer = obj.numVoxelLayers # can be reset in simulation
            self.prev_opt         = None
            self.history          = [] # From previous layer simulation in timely order
            if init_method.startswith('N=') and init_method.strip('N=').isdecimal(): self.N = int(init_method.strip('N='))
            elif init_method == 'constant':                                          self.N = 1
            elif init_method == 'fd':                                                self.N = 2
            elif init_method == 'zero':                                              self.N = 0
            else: raise Exception("Unknown initialization method!")
        def initForFullObject(self):
            if (not self.obj.zeroInit) and (self.prev_opt is not None): return self.prev_opt
            return np.zeros((self.tps.numNodes(), self.obj.DIM))
        def addToHistory(self, x): # from old to new
            if self.N == 0: return
            while len(self.history) >= self.N: self.history.pop(0) # Remove the oldest record
            self.history.append(x)
        def reset(self, simulations_done=None, processing_layer=None):
            self.history = []
            self.simulations_done = 0 if simulations_done is None else simulations_done
            self.processing_layer = self.obj.numVoxelLayers if processing_layer is None else processing_layer
            self.coeffs = []
        def init_zeros(self):
            return np.zeros((self.tps.numNodes(), self.obj.DIM))
        def init_constant(self):
            guess = np.zeros(tuple(self.tps.NbNodesPerDimension) + (self.obj.DIM,))
            guess[:, 1:self.processing_layer+1] = self.obj.toNodeMultiIndex(self.history[-1])[:, 1:self.processing_layer+1]
            return self.obj.toNodeFlatIndex(guess)
        def init_fd(self):
            guess = np.zeros(tuple(self.tps.NbNodesPerDimension) + (self.obj.DIM,))
            guess[:, 1:self.processing_layer+1] = (2 * self.obj.toNodeMultiIndex(self.history[-1]) - self.obj.toNodeMultiIndex(self.history[-2]))[:, 1:self.processing_layer+1]
            return self.obj.toNodeFlatIndex(guess)
        @benchmarkit
        def init_subspace_searching(self, N=None):
            if N is None: N = self.N
            l = self.processing_layer
            benchmark.start_timer_section("U copy")
            U_copy = np.array([self.history[i] for i in range(-N, 0)])
            benchmark.stop_timer_section("U copy")
            U = []
            if self.enrich_mode is None or (self.enrich_mode == 'duplicated' and l == 1):
                U = U_copy
            else:
                for i, u in enumerate(U_copy):
                    u = self.obj.toNodeMultiIndex(u)
                    u1 = np.zeros(tuple(tps.NbNodesPerDimension) + (self.obj.DIM,))
                    u2 = u1.copy()
                    if self.enrich_mode == 'duplicated':
                        u1[:, l:l+1] = u[:, l:l+1]
                        u2[:, 1:l] = u[:, 1:l]
                    else: raise Exception("enrich mode undefined")
                    u1, u2 = self.obj.toNodeFlatIndex(u1), self.obj.toNodeFlatIndex(u2)
                    U.extend([u1, u2])
            benchmark.start_timer_section("KU")
            KU = [self.tps.applyK(u) for u in U]
            benchmark.stop_timer_section("KU")
            benchmark.start_timer_section("U ravel")
            KU_flat = np.array([Ku.ravel() for Ku in KU]).T # numNodes x N
            U_flat_T = np.array([u.ravel() for u in U]) # N x numNodes
            benchmark.stop_timer_section("U ravel")
            benchmark.start_timer_section("Compute U^T KU")
            if self.minimize == 'energy':   A, b = U_flat_T @ KU_flat, U_flat_T.dot(self.tps.buildLoadVector().ravel())# A @ B
            if self.minimize == 'residual': A, b = KU_flat.T @ KU_flat, KU_flat.T.dot(self.tps.buildLoadVector().ravel())
            benchmark.stop_timer_section("Compute U^T KU")
            #c, _, r, *_ = np.linalg.lstsq(A, b, rcond=-1) # rcond=-1 (epsilon), rcond=default(epsilon*max(M,N))
            benchmark.start_timer_section("Solve for coeff")
            c, *_ = sp.linalg.lstsq(A, b) # cond (epsilon * max(sigmas))
            benchmark.stop_timer_section("Solve for coeff")
            guess = self.obj.toNodeFlatIndex(U_flat_T.T.dot(c))
            if self.obj.monitorVars['coeffs']: self.obj.convergenceMonitor.coeffs.append(c)
            return guess
        def nextGuess(self):
            '''
            Support two guess methods and whether to copy values from layer l to layers l+1, l+2, ...

            Parameters
            ----------
            constant:  u^L_i = 0, u^l_i = u^(l+1)
            fd:   u^L_i = 0, u^(L-1)_i = u^L, u^l_i = 2 * u^(l+1) - u^(l+2)
            N: Search the best estimate over the subspace of previous N solutions'''
            if self.simulations_done == 0:
                guess = self.initForFullObject()
            elif self.init_method == 'zero':
                guess = self.init_zeros()
            elif self.init_method.startswith('N='):
                guess = self.init_subspace_searching(min(self.N, self.simulations_done))
            elif self.init_method == 'fd' and self.simulations_done >= 2:
                guess = self.init_fd()
            elif self.init_method == 'zero':
                guess = np.zeros((self.tps.numNodes(), self.obj.DIM))
            else: guess = self.init_constant()
            self.simulations_done += 1
            return guess
    # compute elastic energy: 1/2 u^T K u - u^T g
    def elasticEnergy(self, tps, u, f): return u.ravel().dot(tps.applyK(u).ravel()) / 2 - u.ravel().dot(f.ravel())

    def numReflectedCopies(self): return symmetry_utils.numReflectedCopies(self.symmetry_axes_mask)
    def compliancePerLayer(self, u, f):
        return 0.5 * u.ravel().dot(f.ravel())
    # Derivative of 1/2 f^l . u^l
    def gradientPerLayer(self, u):
        return self.layerByLayerSimulator.complianceGradient(u)

    # Wrappers for compatibility with fd_validation.py; object must have been constructed
    # with `fd_validation_support = True` to use these...
    def numVars(self): return self.vars.size
    def getVars(self): return self.vars

    @benchmarkit_customname('LayerByLayerObjective setVars')
    def setVars(self, x, skipLayers=1, useCPP=True, verbose=False, lblCallback=None): # simulate the full object only
        self.initializer.reset()
        self.setDensities(x)
        self.simulate(increment=skipLayers, useCPP=useCPP, verbose=verbose, lblCallback=lblCallback)
    def energy(self):
        return self.totalCompliances * self.numReflectedCopies()
    def gradient(self):
        result = self.totalComplianceGradients
        if (self.isDownsampling()):
            result = self.highresSimulator.upsampleDensityGradientFrom(self.layerByLayerSimulator, result) # upsample
        return result * self.numReflectedCopies()

    class ConvergenceMonitor:
        coeffs = []
        residualNorms = [] # residualNorms (for initial guess)
        errorNorms = [] # errorNorms (for initial guess)
        residualsPerLayer, displacementsPerLayer = [], [] # plot, visualization
        numCGs, numCGsPerLayer = [], 0 # numCGs
        densityChange = [] # total density on top solid layer / total density
        timestampsPerLayer = [] # plot
        iterationsPerLayer = [] # plot
        orderedLayers = {}
        densities = []

        def __init__(self, obj): # flag is a list of names of vars to save
            self.obj = obj
            self.tps = obj.layerByLayerSimulator
            self.layer = obj.numVoxelLayers
            self.flags = obj.monitorVars
            self.mgType = obj.mgType
            if (self.mgType == LayerByLayerObjective._MGType.AMG) and self.flags['visualization']: raise Exception("AMG does not support visualization.")
            if self.flags['visualization']: self.residuals, self.displacements = [], [] # visualization
            if self.flags['plot']:
                self.colors = plt.get_cmap('jet')(np.linspace(0, 1.0, self.layer))
                self.figure, self.axis = plt.subplots(2, 1)
                self.axis[0].set_xticks(np.arange(20))
                self.axis[0].set_ylabel('Residual Norm')
                self.axis[0].set_yscale('log')
                self.axis[1].set_xticks(np.arange(20))
                self.axis[1].set_ylabel('Time')
        def record(self, i, x, r):
            if self.flags['numCGs']: self.numCGsPerLayer = i # increment by 1 in MG and by 5 in AMG
            if self.flags['visualization']:
                self.residualsPerLayer.append(r)
                self.displacementsPerLayer.append(x)
            elif self.flags['plot']:
                self.residualsPerLayer.append(r)
            if self.flags['plot']:
                self.timestampsPerLayer.append(time.time())
                self.iterationsPerLayer.append(i) # every cg ite for MG (1, 2, .., N) and every 5th for AMG (0, 5, 10, ..., 5*k, N)
        def reset(self, l, u_guess=None, f=None, densities=None):
            self.layer = l
            mg =self.obj.layerByLayerMGSolver
            if self.flags['numCGs']: self.numCGsPerLayer = 0
            if self.flags['residualNorms']: self.residualNorms.append(np.linalg.norm(mg.computeResidual(0, u_guess, f)))
            if self.flags['plot'] and self.flags['visualization']:
                self.residualsPerLayer, self.timestampsPerLayer, self.iterationsPerLayer, self.displacementsPerLayer = [mg.computeResidual(0, u_guess, f)], [time.time()], [0], [u_guess]
            elif self.flags['visualization']:
                self.residualsPerLayer, self.displacementsPerLayer = [mg.computeResidual(0, u_guess, f)], [u_guess]
            elif self.flags['plot']:
                if self.mgType == LayerByLayerObjective._MGType.MG:
                    self.residualsPerLayer, self.timestampsPerLayer, self.iterationsPerLayer = [mg.computeResidual(0, u_guess, f)], [time.time()], [0]
                else: self.residualsPerLayer, self.timestampsPerLayer, self.iterationsPerLayer = [], [], []
            if self.flags['densities']: self.densities.append(densities)
        def postprocess(self, u=None, u_guess=None, f=None):
            if self.flags['numCGs']: self.numCGs.append(self.numCGsPerLayer)
            if self.flags['errorNorms']: self.errorNorms.append(np.linalg.norm(u - u_guess))
            if self.flags['plot'] or self.flags['visualization']:
                f_norm = np.linalg.norm(f)
                if self.flags['plot']:
                    residual_norms = np.linalg.norm(np.array(self.residualsPerLayer), axis=(1,2))/f_norm if self.mgType == LayerByLayerObjective._MGType.MG else self.residualsPerLayer
                    self.timestampsPerLayer = [timestamp - self.timestampsPerLayer[0] for timestamp in self.timestampsPerLayer]
                    color=self.colors[self.layer - 1]
                    self.axis[0].plot(self.iterationsPerLayer, residual_norms, label=f'Num of layers {self.layer}', color=color)
                    self.axis[0].annotate(self.layer, xy=(self.iterationsPerLayer[-1], residual_norms[-1]), xytext=(self.iterationsPerLayer[-1], residual_norms[-1]*0.5), size=7)
                    self.axis[1].plot(self.iterationsPerLayer, self.timestampsPerLayer, label=f'Num of layers {self.layer}', color=color)
                    self.axis[1].annotate(self.layer, xy=(self.iterationsPerLayer[-1], self.timestampsPerLayer[-1]), xytext=(self.iterationsPerLayer[-1], self.timestampsPerLayer[-1]), size=7)
                    self.orderedLayers[self.layer] = (self.iterationsPerLayer[-1], residual_norms[-1], self.timestampsPerLayer[-1])
                if self.flags['visualization']:    # Do not call this on high resolution problems!
                    self.residuals.append([r/f_norm for r in self.residualsPerLayer])
                    self.displacements.append(self.displacementsPerLayer)

        def sortedConvergenceOrderListByCG(self): # (layer, (ite, res, time))
            #return sorted(self.orderedLayers.items(), key=lambda item: item[1][0])
            return [i[0] for i in sorted(self.orderedLayers.items(), key=lambda item: item[1][0])]
        def sortedConvergenceOrderListByTime(self):
            return [i[0] for i in sorted(self.orderedLayers.items(), key=lambda item: item[1][-1])]
        def avgNumCGiterationsPerLayer(self): return np.array(list(self.orderedLayers.values())).mean(0)[0]
        def avgTime(self): return np.array(list(self.orderedLayers.values())).mean(0)[-1]
        def info(self):
            return f"Avg Num CGs: {self.avgNumCGIterations()}\n \
                Layers ordered by number of CG iterations from few to more: {self.sortedConvergenceOrderListByCG()}\n \
                Avg Time: {self.avgTime()}\n \
                Layers ordered by convergence time from low to high: {self.sortedConvergenceOrderListByTime()}"


def initTPS(gridDims, maxCorner=None, densities=None, nu=None):
    dim = len(gridDims)
    size = np.prod(gridDims)
    if maxCorner is None: maxCorner = [d//gridDims[-1] for d in gridDims]
    tps = pyVoxelFEM.TensorProductSimulator([1]*dim, [[0]*dim, maxCorner], gridDims)
    if nu is not None:
        e = tps.ETensor
        e.setIsotropic(1, float(nu)) # Young's modulus=1, Poisson=0.3
        tps.ETensor = e
    if densities is None: tps.setDensities(np.ones(size) * 0.6) # Uniform density
    elif len(densities) == size: tps.setDensities(densities)
    else: tps.setDensitiesFromCoarseGrid(int((size/len(densities))**(1/dim)+0.1), densities)
    return tps

def testConvergence(tps, method, tol, cutCell, uniform):
    import os
    prefix = 'uniform_Emin_1e-3' if uniform else 'nonuniform_Emin_1e-3'
    path = f"benchmark_out/taskA/{prefix}/{method}/{tol}"
    suffix = 'cutcell' if cutCell.lower() == 'true' else 'fullgrid'
    os.makedirs(path, exist_ok=True)
    obj = LayerByLayerObjective(tps, init_method=method, monitorVars={'numCGs':True, 'plot':True, 'coeffs':True}, pcgVars={'tol':float(tol), 'maxIter':500, 'fullMultigrid':False})
    obj.cutCell = (cutCell.lower() == 'true')
    gridDims = str(tps.NbElementsPerDimension).strip('[]').replace(' ', ':')
    if method.startswith('N='):
        write_benchmark_report('', f"{path}/coeff_{gridDims}_{method}_{suffix}.txt") # clear the file
    write_benchmark_report('', f"{path}/cgs_{gridDims}_{method}_{suffix}.txt")

    obj.setVars(obj.layerByLayerSimulator.getDensities()) # choose to skip `increment` layers for next simulation
    # obj.simulate(silence=silence)
    plt.savefig(f"{path}/residual_time_{gridDims}_{method}_{suffix}.png")
    plt.clf()
    data = np.array(obj.convergenceMonitor.numCGs)
    plt.hist(data,  edgecolor='r', bins=range(min(data)-2, max(data)+2, 1), align='left', label=data.mean())
    plt.xlabel('Num of CGs')
    plt.ylabel('Num of simulations')
    plt.xticks(range(min(data) - 2, max(data)+2, 1))
    plt.legend()
    plt.savefig(f"{path}/cgs_{gridDims}_{method}_{suffix}.png")
    for i, cgs in enumerate(data):
        write_benchmark_report('print(n - i, cgs)', f"{path}/cgs_{gridDims}_{method}_{suffix}.txt", 'a', {'n':tps.NbElementsPerDimension[1], 'i':i, 'cgs':cgs})
    write_benchmark_report("print('Avg', avg)", f"{path}/cgs_{gridDims}_{method}_{suffix}.txt", 'a', {'avg': data.mean()})
    for c in obj.convergenceMonitor.coeffs: write_benchmark_report('print(s)', f"{path}/coeff_{gridDims}_{method}_{suffix}.txt", 'a', {'s': str(c).replace('\n', '')})
    print('Average compliance per layer:', obj.totalCompliances / obj.numVoxelLayers)
    print('Total compliance gradients:', obj.totalComplianceGradients)
    print(np.sum(obj.convergenceMonitor.numCGs)/obj.numVoxelLayers)
    # data = np.load('data/densities2_32:32.npy')
    # data = data.reshape(32, 32)
    # fig, ax =plt.subplots()
    # ax.imshow(data.T, cmap='jet', origin='lower')
    # plt.title("MBB Samples: 32 x 32")
    # plt.xticks(range(32))
    # plt.yticks(range(32))
    # c = n//32
    # for i, y in enumerate(obj.convergenceMonitor.numCGs):
    #     ax.annotate(str(y), xy=(0, 0), xytext=(-5-(c-i%c),32-i//c-1.5), xycoords='data', annotation_clip=False)#ax.get_xaxis_transform())
    # ax.annotate("CGs",  xy=(0, 0), xytext=(-5.5,32), xycoords='data', annotation_clip=False)
    # ax.grid()
    # plt.savefig(f'benchmark_out/poisson_{nu}/tol_{tol}/{density_type}/cgs_{n}_{method}_{suffix}.pdf')
    print('-'*60)
def testGradient(tps, downsampling_levels=0, customArgs=None):
    import fd_validation as fd
    # densities = np.random.rand(n*n) * 0.9
    obj = LayerByLayerObjective(tps, downsampling_levels=downsampling_levels, fd_validation_support=True, pcgVars={'tol': 1e-12})
    fd.setVars(obj, None, customArgs=customArgs) # Use current Vars
    epsilons = np.logspace(-8, -1, 30)
    fd.gradConvergencePlot(obj, epsilons=epsilons, customArgs=customArgs)
    plt.savefig(f'benchmark_out/fd_validation.png')

if __name__ == '__main__':
    import sys
    args = sys.argv
    dims, tol, method, cutCell, uniform = args[1:]
    if uniform.lower() == 'true': densities = None
    else: densities = np.load('data/densities2_32:32.npy')

    tps = initTPS([int(d) for d in dims.split(',')], nu=0.3, densities=densities)
    tps.E_min = 1e-3
    testConvergence(tps, method, tol, cutCell, uniform=(densities is None))
    # density_type = 'uniform' #'densities2_32:32'#
    # i = 44
    # densities = np.load(f'benchmark_out/densities_mma1_10.0_{m}_{n}_data/{i}.npy')
    # testGradient(tps, downsampling_levels=1, customArgs={'skipLayers': 2, 'useCPP': True})
