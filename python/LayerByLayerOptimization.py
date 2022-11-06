from LayerByLayerObjective import *
import benchmark
import sys, os
sys.path.append(os.path.join('.', 'helpers'))
from write_benchmark import write_benchmark_report
import pyOptimizer
import logging
import parallelism, psutil
from tqdm import tqdm
parallelism.set_max_num_tbb_threads(psutil.cpu_count(logical=False))
parallelism.set_gradient_assembly_num_threads(min(psutil.cpu_count(logical=False), 8))

class MMA:
    def __init__(self, prob, maxIter=50, relTol=1e-5, callBeforeSetVars=None, callback=None, callDensity=None):
        self.maxIter = maxIter
        self.m = 1
        self.n = prob.numVars()
        def gradients(x): # (m+1) x n
            result = np.stack([prob.evaluateObjectiveGradient(), -prob.evaluateConstraintsJacobian()[0]])
            return result
        def objAndConstr(x): # called before gradients
            if callBeforeSetVars is not None: callBeforeSetVars()
            prob.setVars(x)
            result = np.stack([prob.evaluateObjective(), -prob.evaluateConstraints()[0]])
            if callDensity is not None: callDensity(prob.getDensities())
            if callback is not None: callback(prob.getDensities(), result[0], -result[1])
            return result
        self.mma = pyOptimizer.MMA(self.n, 1, np.zeros(self.n), np.ones(self.n), objAndConstr, gradients)
    @benchmarkit_customname("MMA optimization")
    def run(self, x0, showProgress=False):
        # self.mma.enableGCMMA(True)
        self.mma.setInitialVar(x0)
        if showProgress:
            for _ in tqdm(range(self.maxIter)):
                self.mma.step()
            return
        for _ in range(self.maxIter):
            self.mma.step()

class CGRecord:
    def __init__(self):
        self.cgCounter = 0
        self.numCGs = []
    def reset(self):
        self.cgCounter = 0
        self.numCGs = []
    def resetCGCounter(self):
        self.cgCounter = 0
    def record(self, i, r):
        self.cgCounter += 1
    def updateCGs(self):
        self.numCGs.append(self.cgCounter)
    def avgCGs(self): return np.array(self.numCGs).mean()

def getClass(baseClass):
    class LayerByLayerOptimizationProblem(baseClass):
        def __init__(self, tps, optObj, constraints, filters, layObj, weight):
            super().__init__(tps, optObj, constraints, filters)
            self.optObj = optObj
            self.layObj = layObj
            self.weight = weight
            self.skipLayers = 1
            self.opt_obj = 0
            self.lbl_obj = 0
        def setVars(self, x, verbose=False):
            benchmark.start_timer_section('Optimization setVars')
            super().setVars(x)
            benchmark.stop_timer_section('Optimization setVars')
            self.layObj.setVars(self.getDensities(), self.skipLayers, verbose)
        def evaluateObjective(self):
            self.opt_obj = super().evaluateObjective()
            self.lbl_obj = self.layObj.energy()
            return self.opt_obj + self.weight * self.lbl_obj
        def evaluateObjectiveGradient(self):
            return super().evaluateObjectiveGradient() + self.weight * self.filterChain.backprop(self.layObj.gradient())
        def gradient(self): # For fd validation
            return self.evaluateObjectiveGradient()
        def energy(self): # For fd validation
            return self.evaluateObjective()
    return LayerByLayerOptimizationProblem

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

def writeLogs():
    global gridDims, algorithm, weight, maxCorner, tol, init_method, MATERIAL_PATH, BC_PATH, maxVolume, uniformDensity, filters, tps, optObj, layObj, top
    def log(name, value): return f"{name:<30}{value:<20}\n"
    info = "\n" + "Basic variables\n" + '-'*50 + '\n'
    info += log('maxCorner', str(maxCorner))
    info += log('dimensions', str(gridDims))
    info += log('algorithm', algorithm)
    info += log('weight', weight)
    info += log('LBL CG tol', tol)
    info += log('init method', init_method)
    info += "\n" + "Optimization Simulator\n" + '-'*50 + '\n'
    info += log('material', MATERIAL_PATH)
    info += log('BC', BC_PATH)
    info += log('maxVolume', maxVolume)
    info += log('init density', uniformDensity)
    info += log('E_min', tps.E_min)
    info += log('gamma', tps.gamma)
    global numCoaseningLevels
    info += "\n" + "Optimization objective\n" + '-'*50 + '\n'
    info += log('numCoaseningLevels', numCoaseningLevels)
    info += log('maxCGIter', optObj.cgIter)
    info += log('cgTol', optObj.tol)
    info += log('mgSmoothingIterations', optObj.mgSmoothingIterations)
    info += log('fullMultigrid', optObj.fullMultigrid)
    if layObj is not None:
        info += "\n" + "LayerByLayer objective\n" + '-'*50 + '\n'
        info += log('initFromLastOptimization', not layObj.zeroInit)
        info += log('downsampling levels', layObj.downsampling_levels)
        info += log('E_min', layObj.layerByLayerSimulator.E_min)
        info += log('Inpt law', layObj.intpLaw.name)
        info += log('RAMP q', layObj.layerByLayerSimulator.q)
        info += log('maxCGIter', layObj.pcgVars['maxIter'])
        info += log('cgTol', layObj.pcgVars['tol'])
        info += log('mgSmoothingIterations', layObj.pcgVars['mgSmoothingIterations'])
        info += log('fullMultigrid', layObj.pcgVars['fullMultigrid'])
    info += "\n" + "TO Problem\n" + '-'*50 + '\n'
    if layObj is not None:
        info += log('SkipLayers', top.skipLayers)
    info += log('smooth radius', filters[0].radius)
    info += log('smooth type', str(filters[0].type))
    info += log('projection beta', filters[-1].beta)
    logging.info(info)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Layer-by-layer optimization problem')
    parser.add_argument("gridDims", help="Grid dimensions")
    parser.add_argument("weight", help="Weight of layer-by-layer objective")
    parser.add_argument("--algo", default="mma", help="Algorithm for optimization")
    parser.add_argument("--coarse", default=-1, type=int, help="Number of coarsening levels")
    parser.add_argument("--tol", default="1e-5", help="CG tolerance")
    parser.add_argument("--init", default="N=3", help="Initialization method for constructing initial guess ")
    parser.add_argument("--down", default=0, type=int, help="Downsampling level")
    parser.add_argument("--incr", default=1, type=int, help="Layer increment")
    options = parser.parse_args()

    maxCorner = None
    gridDims  = options.gridDims
    weight    = options.weight
    algorithm = options.algo
    coarse    = options.coarse
    tol       = options.tol
    init_method = options.init
    downsampleLevel = options.down
    skipLayers = options.incr

    # intlaw = 'RAMP'
    bc = 'mbb_N'
    dir_suffix = ''
    if downsampleLevel > 0 and skipLayers == 1: dir_suffix = f"/downsampled_{downsampleLevel}"
    elif downsampleLevel == 0 and skipLayers > 1: dir_suffix = f"/skiplayers_{skipLayers}"
    elif downsampleLevel > 0 and skipLayers > 1: raise Exception("Please specify only one of downsamping or increment.")
    path = f"benchmark_out/{bc}/weight_{weight}/{gridDims.replace(',', '-')}/tol{tol}_{init_method}" + dir_suffix
    os.makedirs(path, exist_ok=True)

    benchmark.reset()

####### Set up simulator
    gridDims = [int(d) for d in gridDims.split(',')]
    DIM = len(gridDims)
    MATERIAL_PATH = os.path.join('..', 'examples', 'materials', 'B9Creator.material')
    if DIM == 3: BC_PATH = os.path.join('..', 'examples', 'bcs', '3D', f'{bc}.bc')
    else: BC_PATH = os.path.join('..', 'examples', 'bcs', f'{bc}.bc')
    maxVolume = 0.6
    constraints = [pyVoxelFEM.TotalVolumeConstraint(maxVolume)]
    filters = [pyVoxelFEM.SmoothingFilter(radius=2, type=pyVoxelFEM.SmoothingFilter.Type.Linear), pyVoxelFEM.ProjectionFilter(beta=5)]
    uniformDensity = filters[-1].invert(maxVolume)
    tps = initTPS(gridDims, maxCorner, densities=np.ones(np.prod(gridDims)) * uniformDensity)
    tps.readMaterial(MATERIAL_PATH)
    tps.applyDisplacementsAndLoadsFromFile(BC_PATH)

######## init optimization objective
    numCoaseningLevels = optMGLevel(tps.NbElementsPerDimension, DIM) if coarse < 0 else coarse # Number of optimal coasening levels
    optObj = pyVoxelFEM.MultigridComplianceObjective(tps.multigridSolver(numCoaseningLevels))
    optObj.mgSmoothingIterations = 1
    optObj.tol = 1e-4
    layObj = None
    LayerByLayerOptimizationProblem = getClass(eval(pyVoxelFEM.getClassName(tps, "TopologyOptimizationProblem")))

    if float(weight) < 1e-5:
        top = pyVoxelFEM.TopologyOptimizationProblem(tps, optObj, constraints, filters)
    else:
    ######## init layer-by-layer objective : do not manually numCoaseningLevels here, let LBL objective find a suitable for its downsampled simulator
        layObj = LayerByLayerObjective(tps, mg_levels=None if coarse < 0 else coarse, init_method=init_method, downsampling_levels=downsampleLevel)
        layObj.setCGTol(float(tol))#
        layObj.zeroInit = False # if false, the full object simulation will use the result of its previous optimization
        # Using SIMP / RAMP
        # layObj.setIntptLaw(eval(f"LayerByLayerObjective._InterpolationLaw.{intlaw}"))
    ######## init Topology Optimziation Problem
        top = LayerByLayerOptimizationProblem(tps, optObj, constraints, filters, layObj, float(weight))
        top.skipLayers = skipLayers # Set to skip some layers for layer-by-layer simulation

############## Log all the settings
    logging.basicConfig(filename=path+'/settings.log', filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
    writeLogs()

    output = [] # record objectives and constrain values at each step
    optCGRecorder = CGRecord()
    layCGRecorder = CGRecord()
    optObj.residual_cb = optCGRecorder.record
    if layObj is not None: layObj.pcgVars['it_callback'] = lambda i, _, r: layCGRecorder.record(i, r)
    def callBeforeSetVars():
        optCGRecorder.resetCGCounter()
        layCGRecorder.resetCGCounter()
    def callback(density, objective, constraint):
        optCGRecorder.updateCGs()
        if layObj is None:
            out = f"Objective {objective}, constraint {constraint}"
        else:
            out = f"Objective [{objective}, {top.opt_obj}, {top.lbl_obj}], constraint {constraint}"
            layCGRecorder.updateCGs()
        print(out)
        output.append(out)

############## Optimize
    max_iters, ftol_rel = 50, 1e-12
    if algorithm.upper() not in ['MMA', 'MMA2', 'OC']: raise Exception('Unknown algorithm!')
    opt = eval(algorithm.upper())(top, max_iters, ftol_rel, callBeforeSetVars, callback)
    opt.run(tps.getDensities())

    write_benchmark_report('benchmark.report()', path=path+'/benchmark_report.txt', msg='Benchmark report', globalVars={'benchmark': benchmark})

####### Collect data
    ##### Final density #####
    densities = top.getDensities()
    vars = top.getVars()
    np.save(f"{path}/finalDensity.npy", densities)
    np.save(f"{path}/finalVars.npy", vars)

    ##### CGs #####
    opt_cgs, tol_cgs = 0, 0
    with open(path+'/cgs.txt', 'w') as f:
        for i, (opt, lay) in enumerate(zip(optCGRecorder.numCGs, layCGRecorder.numCGs)):
            opt_cgs += opt
            tol_cgs += lay/gridDims[1]
            f.writelines(f"{i} {opt} {lay/gridDims[1]}\n")
        f.write(f"Avg {opt_cgs/max_iters}, {tol_cgs/max_iters}")

    with open(path+'/objective.txt', 'w') as f:
        for i, line in enumerate(output):
            f.write(f"{i}, {line}")
        # f.write("Final objectives\n")
        # f.write(f"Objective {finalObjective}, constraint {objConstrPair[1]}\n")
    ##### Skewness of design ###### sum_i rho_i * (1 - rho_i)
    skewness = (top.getDensities() * (1 - top.getDensities())).sum()
    with open(path + '/binariness.txt', 'w') as f:
        f.write(f"{skewness}\n")
