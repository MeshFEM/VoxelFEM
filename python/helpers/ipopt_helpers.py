import numpy as np
import pyVoxelFEM
from history_helpers import optimizationHistory

def initializeTensorProductSimulator(orderFEM, domainCorners, numberElements, uniformDensity, E0, Emin, SIMPExp, materialPath, bcsPath):
    TPS = pyVoxelFEM.TensorProductSimulator(orderFEM, domainCorners, numberElements)
    TPS.readMaterial(materialPath)
    TPS.setUniformDensities(uniformDensity)
    TPS.applyDisplacementsAndLoadsFromFile(bcsPath)
    TPS.E_0 = E0
    TPS.E_min = Emin
    TPS.gamma = SIMPExp
    return TPS

def initializeIpoptProblem(TOP, previousHistory=[], recording = True):
    "Wrap TopologyOptimizationProblem into problem needed by ipopt."
    import cyipopt as ipopt
    
    constraintsNumber = 1
    numVars = TOP.numVars()
    
    # Variables bounds (\in [0, 1])
    lb = np.zeros(numVars)
    ub = np.ones(numVars)

    # Constraints bounds (all the constraints are imposed positive)
    cl = [0.0] * constraintsNumber
    cu = None

    # Wrap the TopologyoptimizationProblem so that the interface complies with ipopt optimizer
    problemObj = problemObjectWrapper(TOP, previousHistory)
    problemObj.setRecording(recording)
    
    # Initialize the non linear problem on which optimization is run
    nlp = ipopt.problem(
        n=numVars,
        m=constraintsNumber,
        problem_obj=problemObj,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    return nlp, problemObj.history    # nlp runs optimization, optimization history stores intermediate data


# Wrapper for the TopologyOptimizationProblem that makes it compatible with ipopt
class problemObjectWrapper:
    def __init__(self, problem, previousHistory=[]):
        if previousHistory == []:
            self.history = optimizationHistory()    # initialize empty memory
        else:
            self.history = previousHistory          # initialize history from backup
        self.problem = problem                      # the problem to be optimized
        self.recordingHistory = True

    def objective(self, x):
        self.problem.setVars(x)
        return self.problem.evaluateObjective()

    def gradient(self, x):
        self.problem.setVars(x)
        return self.problem.evaluateObjectiveGradient()

    def constraints(self, x):
        self.problem.setVars(x)
        return self.problem.evaluateConstraints()

    def jacobian(self, x):
        self.problem.setVars(x)
        return self.problem.evaluateConstraintsJacobian()

    def setRecording(self, onoff):
        self.recordingHistory = onoff

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
            mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        # Print initial info every time the optimizer is restarted
        if iter_count == 0:
            print('Running with:')
            for constr in self.problem.constraints:
                if isinstance(constr, pyVoxelFEM.TotalVolumeConstraint):
                    print(" Total volume constraint (fraction={:1.1f})".format(constr.volumeFraction))
            for filt in self.problem.filters:
                if isinstance(filt, pyVoxelFEM.SmoothingFilter):
                    print(" Smoothing filter        (radius={:1.0f})".format(filt.radius))
                elif isinstance(filt, pyVoxelFEM.ProjectionFilter):
                    print(" Projection filter       (beta={:1.0f})".format(filt.beta))
                elif isinstance(filt, pyVoxelFEM.LangelaarFilter):
                    print(" Langelaar filter")
        # Print info
        print("Iter: {:2.0f}   ".format(self.history.recordedEpochs+1), end='')
        print("Obj: {:4.3f}   ".format(obj_value))
        # Record current values in history
        if (self.recordingHistory):
            self.history.update(
                self.problem.getDensities(),
                obj_value
            )
