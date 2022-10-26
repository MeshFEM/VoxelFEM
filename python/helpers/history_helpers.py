import numpy as np
import matplotlib.pyplot as plt
from multiscale_helpers import upscaleScalarField

# Optimiation history recording info at each iteration
class optimizationHistory:
    
    def __init__(self):
        self.recordedEpochs = 0
        self.density = []
        self.iter = []
        self.objective = []
        self.nondiscreteness = []
        
    def update(self, x, obj):
        self.recordedEpochs += 1
        self.density.append(x)
        self.iter.append(self.recordedEpochs)
        self.objective.append(obj)
        self.nondiscreteness.append(nondiscreteness(self.density[-1]))
        
    def subsample(self, period):
        subsampledHistory = copy.deepcopy(self)
        sampler = np.arange(0, self.recordedEpochs, period)
        if sampler[-1] != self.recordedEpochs-1:  # always keep the last iteration
            sampler = np.append(sampler, self.recordedEpochs-1)
        subsampledHistory.density = [self.density[i] for i in sampler]
        subsampledHistory.iter = [self.iter[i] for i in sampler]
        subsampledHistory.objective = [self.objective[i] for i in sampler]
        subsampledHistory.nondiscreteness = [self.nondiscreteness[i] for i in sampler]
        subsampledHistory.recordedEpochs = len(subsampledHistory.density)
        return subsampledHistory
    
    def plotObjective(self, ax=None, plotLegend=''):
        if ax == None:
            plt.semilogy(np.arange(self.recordedEpochs), self.objective)
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
        else:
            ax.semilogy(np.arange(self.recordedEpochs), self.objective, label=plotLegend)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Objective')
        
    def plotNondiscreteness(self, ax=None, plotLegend=''):
        if ax == None:
            plt.plot(np.arange(self.recordedEpochs), self.nondiscreteness)
            plt.ylim([0, 1])
            plt.xlabel('Iteration')
            plt.ylabel('Non-discreteness')
        else:
            ax.plot(np.arange(self.recordedEpochs), self.nondiscreteness, label=plotLegend)
            ax.set_ylim([0, 1])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Non-discreteness')

        
def nondiscreteness(density):
    """
    Measure sharpness of a density field.
    Result is in [0, 1], solid and void voxels do not contribute.
    """
    return np.sum(4*density*(1-density))/np.size(density)

    
def upscaleHistory(history, gridDimensions):
    """
    Double the number of elements in each of the coordinate direction
    for each of the densities recorded in history.
    """
    for i in range(history.recordedEpochs):
        _, history.density[i] = upscaleScalarField(gridDimensions, history.density[i])
    return history