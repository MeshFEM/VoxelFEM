import MeshFEM
import tri_mesh_viewer, mesh
import density_tools
import matplotlib
import numpy as np


UCDBlue = matplotlib.colors.LinearSegmentedColormap('UCDavisBlue', segmentdata={'red': [[0, 1, 1], [1, 0.00784314, 0.00784314]],
                                                                                'green': [[0, 1, 1], [1, 0.15686275, 0.15686275]],
                                                                                'blue':  [[0, 1, 1], [1, 0.31764706, 0.31764706]]}, N=256)
NYUPurple = matplotlib.colors.LinearSegmentedColormap('NYUPurple',   segmentdata={'red': [[0, 1, 1], [1, 0.34117647, 0.34117647]],
                                                                                  'green': [[0, 1, 1], [1, 0.02352941, 0.02352941]],
                                                                                  'blue':  [[0, 1, 1], [1, 0.54901961, 0.54901961]]}, N=256)
Black = matplotlib.colors.LinearSegmentedColormap('BlackWhite', segmentdata={'red': [[0, 1, 1], [1, 0, 0]],
                                                                                'green': [[0, 1, 1], [1, 0, 0]],
                                                                                'blue':  [[0, 1, 1], [1, 0, 0]]}, N=256)

Green = matplotlib.colors.LinearSegmentedColormap('NYUPurple',   segmentdata={'red': [[0, 1, 1], [1, 0.20784314, 0.20784314]],
                                                                                  'green': [[0, 1, 1], [1, 0.36862745, 0.36862745]],
                                                                                  'blue':  [[0, 1, 1], [1, 0.23137255, 0.23137255]]}, N=256)
CelticsGreen = matplotlib.colors.LinearSegmentedColormap('CelticsGreen', segmentdata={'red': [[0, 1, 1], [1, 0, 0]],
                                                                                'green': [[0, 1, 1], [1, 122/256, 122/256]],
                                                                                'blue':  [[0, 1, 1], [1, 51/256, 51/256]]}, N=256)
DefaultCMap = CelticsGreen


class TPSViewer(tri_mesh_viewer.TriMeshViewer):
    """
    Visualize a 2D or 3D `TensorProductSimulator` object as deformable voxel grid.
    """
    def __init__(self, tps, colormap=DefaultCMap, **viewer_kwargs):
        self.tps = tps
        self._density_threshold = None
        self._density_colormap = colormap

        if 'scalarField' in viewer_kwargs:
            raise Exception('Manual scalar fields are not supported')

        m = mesh.VoxelBoundaryMesh(tps.gridShape, tps.dx)
        super().__init__(m, **viewer_kwargs)

    @property
    def densityThreshold(self): return self._density_threshold

    @densityThreshold.setter
    def densityThreshold(self, val):
        # When masking out low-density voxels, calls to `update()` will have a
        # distracting flicker when the old mesh is removed and the new one is
        # added (since the mesh connectivity changes in this case). We suppress
        # this flicker using the `avoidRedrawFlicker` functionality, but
        # disable this functionality when unnecessary since it causes choppy
        # mouse interaction.
        self.avoidRedrawFlicker = val is not None

        if (val is None) and (self._density_threshold is not None):
            # Disabling the thresholding requires switching to an unmasked VoxelBoundaryMesh...
            self.mesh = mesh.VoxelBoundaryMesh(self.tps.gridShape, self.tps.dx)

        self._density_threshold = val

        self.update()

    def applyDisplacements(self, u, scale = 1.0):
        """
        Visualize the current `tps` geometry deformed according to displacement field `scale * u`.
        """
        if (u.shape[1] == 2):
            u = np.pad(u, [(0, 0), (0, 1)])
        self.update(displacementField=scale * u)

    def getVisualizationGeometry(self):
        if self._density_threshold is not None:
            tps = self.tps
            mask = tps.getDensities().reshape(tps.gridShape) >= self._density_threshold
            self.mesh = mesh.VoxelBoundaryMesh(tps.gridShape, tps.dx, mask)
        return self.mesh.visualizationGeometry()

    @property
    def densityColormap(self): return self._density_colormap

    @densityColormap.setter
    def densityColormap(self, cmap):
        self._density_colormap = cmap
        self.update()

    def update(self, preserveExisting=False, mesh=None, **kwargs):
        if 'scalarField' in kwargs and kwargs['scalarField'] is not None:
            raise Exception('Manual scalar fields are not supported')
        kwargs['scalarField'] = self._densityScalarField()
        super().update(preserveExisting=preserveExisting, mesh=mesh, **kwargs)

    def _densityScalarField(self):
        return {'data': self.tps.getDensities(), 'colormap': self._density_colormap, 'vmin': 0.0, 'vmax': 1.0}

class ContourViewer(tri_mesh_viewer.TriMeshViewer):
    """
    Visualize a 3D `TensorProductSimulator` as a triangle mesh of the `rho=1/2` contour.
    """
    class MeshWrapper:
        def __init__(self, tps, m):
            self.mesh = m
            self.tps = tps

        def visualizationField(self, u):
            return np.array(self.tps.sampleNodalField(u, self.mesh.vertices()), dtype=np.float32)

        def visualizationGeometry(self):
            return self.mesh.visualizationGeometry()

    def __init__(self, tps, **viewer_kwargs):
        self.tps = tps
        dim = len(tps.gridShape)
        if dim != 3: raise Exception('Only implemented for 3D grids')
        super().__init__(None, **viewer_kwargs)

        self.avoidRedrawFlicker = True # The mesh connectivity generally changes with each update, leading to flickering :(


    def setGeometry(self, *args, **kwargs):
        umm = kwargs['updateModelMatrix']
        kwargs['updateModelMatrix'] = False # work around failure happening with an empty contour
        super().setGeometry(*args, **kwargs)
        if umm:
            # set the model matrix based on the domain's bounding box (rather than the contour, which may be empty)
            vertices = np.array(self.tps.domain)
            center = np.mean(vertices, axis=0)
            bbSize = np.max(np.abs(vertices - center))
            scaleFactor = 2.0 / bbSize
            quaternion = [0, 0, 0, 1]
            self.transformModel(-scaleFactor * center, scaleFactor, quaternion)

    def update(self, *args, **kargs):
        if (not ('mesh' in kargs)): self.mesh = self._getDensityContourMesh()
        super().update(*args, **kargs)

    def applyDisplacements(self, u, scale = 1.0):
        if (u.shape[1] == 2):
            u = np.pad(u, [(0, 0), (0, 1)])
        self.update(displacementField=scale * u)

    def _getDensityContourMesh(self):
        tps = self.tps
        return ContourViewer.MeshWrapper(tps, mesh.Mesh(*density_tools.contour(tps.getDensities().reshape(tps.gridShape), tps.dx)))

def sliceVideo(path, densities, size=6, display=True, axis=1, vmin=0, vmax=1):
    """
    A video showing each layer slice from bottom to top.
    We assume a printing direction along the Y (second) axis.
    """
    from matplotlib import pyplot as plt
    import matplotlib.animation
    aspect = densities.shape[0] / densities.shape[2]
    # fig = plt.figure(figsize=(size * aspect, size))
    fig = plt.figure()
    ax = plt.gca()
    nframes=densities.shape[1]
    def frame(i):
        ax.cla()
        slc = [slice(None)] * 3
        slc[axis] = i
        ax.imshow(densities[tuple(slc)].T, vmin=vmin, vmax=vmax, origin='lower')
        plt.axis('off')
        plt.tight_layout(pad=0)
    ani = matplotlib.animation.FuncAnimation(fig, frame, frames=nframes)
    ani.save(path, fps=30)
    plt.close()
    if display:
        from IPython.display import Video
        return Video(path)

def viewBC(DIM, path, output=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig = plt.figure()
    if DIM == 2:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        w, h = 2, 1
        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(-0.1, 1.1)
        domain = mpatches.Rectangle((0, 0), 2, 1, color=(0.8, 0.8, 0.8, 0.5))
        ax.add_patch(domain)
        with open(path, "r") as f:
            for l in f.readlines():
                if "dirichlet" in l:
                    minCorner = [float(s.strip('[],')) for s in l.split()[11:14]]
                    maxCorner = [float(s.strip('[],')) for s in l.split()[15:18]]
                    for i in range(DIM):
                        if minCorner[i] < 0: minCorner[i] = min(-0.01, minCorner[i])
                        else: minCorner[i] = min(0.99, minCorner[i])
                        if maxCorner[i] < 1: maxCorner[i] = max(0.01, maxCorner[i])
                        else: maxCorner[i] = max(1.01, maxCorner[i])
                    ax.add_patch(mpatches.Rectangle((minCorner[0]*w, minCorner[1]*h), (maxCorner[0]-minCorner[0])*w, (maxCorner[1]-minCorner[1])*h, color=(1, 1, 0, 1)))
                elif "force" in l:
                    minCorner = [float(s.strip('[],')) for s in l.split()[11:14]]
                    maxCorner = [float(s.strip('[],')) for s in l.split()[15:18]]
                    force = [float(s.strip('[],')) for s in l.split()[4:7]]
                    x, y = np.meshgrid(np.arange(minCorner[0], maxCorner[0], 0.1), np.arange(minCorner[1], maxCorner[1], 0.1))
                    ax.quiver(x.ravel() * w, y.ravel() * h, [force[0]]*len(x), [force[1]]*len(x), color='red')
    elif DIM == 3:
        ax = fig.add_subplot(111, projection='3d')
        def plot_cube(minCorner, maxCorner, color=(0.8, 0.8, 0.8), solid=False):
            x, y = np.meshgrid([minCorner[0], maxCorner[0]], [minCorner[2], maxCorner[2]])
            ax.plot_surface(x, y, np.ones_like(x) * minCorner[1], color=color, alpha=0.2 if not solid else 1)
            ax.plot_surface(x, y, np.ones_like(x) * maxCorner[1], color=color, alpha=0.2 if not solid else 1)
            x, z = np.meshgrid([minCorner[0], maxCorner[0]], [minCorner[1], maxCorner[1]])
            ax.plot_surface(x, np.ones_like(x) * minCorner[2], z, color=color, alpha=0.1 if not solid else 1)
            ax.plot_surface(x, np.ones_like(x) * maxCorner[2], z, color=color, alpha=0.2 if not solid else 1)
            y, z = np.meshgrid([minCorner[2], maxCorner[2]], [minCorner[1], maxCorner[1]])
            ax.plot_surface(np.ones_like(x) * minCorner[0], y, z, color=color, alpha=0.2 if not solid else 1)
            ax.plot_surface(np.ones_like(x) * maxCorner[0], y, z, color=color, alpha=0.4 if not solid else 1)
        X, Y, Z = 2, 1, 1
        plot_cube([0, 0, 0], [X, Z, Y])
        ax.set_box_aspect((X, Z, Y))
        with open(path, "r") as f:
            for l in f.readlines():
                if "dirichlet" in l:
                    minCorner = [float(s.strip('[],')) for s in l.split()[11:14]]
                    maxCorner = [float(s.strip('[],')) for s in l.split()[15:18]]
                    for i in range(DIM):
                        if minCorner[i] < 0: minCorner[i] = min(-0.01, minCorner[i])
                        else: minCorner[i] = min(0.99, minCorner[i])
                        if maxCorner[i] < 1: maxCorner[i] = max(0.01, maxCorner[i])
                        else: maxCorner[i] = max(1.01, maxCorner[i])
                    plot_cube([a*b for (a, b) in zip(minCorner, [X, Y, Z])], [a*b for (a, b) in zip(maxCorner, [X, Y, Z])], color=(1, 1, 0), solid=True)
                elif "force" in l:
                    minCorner = [float(s.strip('[],')) for s in l.split()[11:14]]
                    maxCorner = [float(s.strip('[],')) for s in l.split()[15:18]]
                    force = [float(s.strip('[],')) for s in l.split()[4:7]]
                    x, y, z = np.meshgrid(np.arange(minCorner[0], maxCorner[0], 0.1), np.arange(minCorner[2], maxCorner[2], 0.1), np.arange(minCorner[1], maxCorner[1], 0.1))
                    ax.quiver(x.ravel() * X, y.ravel() * Y, z.ravel() * Z, [force[0]*0.2]*len(x), [force[2]*0.2]*len(x), [force[1]*0.2]*len(x), color='red')
    ax.axis('off')
    plt.show()
    if output is not None: fig.savefig(output, bbox_inches='tight')

def vectorFieldVis(numNodesPerDim, field, scale=None, keyscale=None, scale_units=None, show=False, savePath=None, title=None):
    if len(numNodesPerDim) > 2:
        print('3D or higher dimension not supported yet.')
    import matplotlib.pyplot as plt
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
    import matplotlib.pyplot as plt
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
