import matplotlib.pyplot as plt
import numpy as np

def numericalDerivative(F, x, h, direction):
    "Second order centered differences numerical derivative"
    return (F(x + h * direction) - F(x - h * direction)) / (2 * h)

def plotConvergence(F, x, increments, direction, referenceValue, title=""):
    "Create a loglog error plot from derivatives data."
    numerical = []
    for h in increments:
        numerical.append(numericalDerivative(F, x, h, direction))
    plt.figure()
    errors = np.abs(np.array(numerical) - referenceValue)
    plt.loglog(increments, errors)
    plt.loglog(increments, np.array(increments)**2)
    plt.legend(['Error', r'$h^2$'])
    plt.title(title)
    plt.xlabel("h")
    plt.ylabel("Error")
    plt.show()
