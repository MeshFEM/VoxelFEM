import numpy as np
import matplotlib.pyplot as plt

def vectorFieldVis(numNodesPerDim, field, scale=None, keyscale=None, scale_units=None, show=False, savePath=None, title=None):
    if len(numNodesPerDim) > 2:
        print('3D or higher dimension not supported yet.')
    DIM = 2
    nodeDims = tuple(numNodesPerDim) + (DIM,)
    x, y = np.meshgrid(range(nodeDims[0]), range(nodeDims[1]), indexing='ij')
    if len(field.shape) == 2:
        field = np.reshape(field, nodeDims)
    u, v = field[..., 0], field[..., 1]
    plt.figure()
    Q = plt.quiver(x, y, u, v, color='b', units='xy', scale=scale, scale_units=scale_units)
    if keyscale is not None:
        plt.quiverkey(Q, 0.9, 0.9, keyscale,  f'{keyscale}', color='r', coordinates='figure',linewidth=1)
    plt.title(title)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    if show: 
        plt.show()
    if savePath is not None:
        plt.savefig(savePath)
    plt.close()
        
def colorVis(numNodesPerDim, field, savePath, update=None, title=None, label=None, vmin=None, vmax=None):
    if len(numNodesPerDim) > 2:
        print('3D or higher dimension not supported yet.')
    field = field.reshape(numNodesPerDim)
    if title is not None: plt.title(title)
    if update is None:
        xtick, ytick = np.arange(int(numNodesPerDim[0])), np.arange(int(numNodesPerDim[1]))
        plt.xticks(xtick)
        plt.yticks(ytick)
        update = plt.imshow(field.T, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(label=label, orientation="vertical")
    else: update.set_data(field.T)
    plt.savefig(savePath)
    return update

if __name__ == '__main__':
    dims = (50, 50)
    data = np.random.random(dims + (2,))
    # vecdectorFieldVis(dims, data)
    #plt.imshow(data, origin='lower')
    #plt.xticks(np.arange(0, 50, 10), np.arange(0, 50, 10)+10)
    #plt.figure()
    #data = np.zeros((50,50))
    #plt.imshow(data, origin='lower')
    #plt.xticks(np.arange(0, 50, 10), np.arange(0, 50, 10)+10)
    #plt.show()