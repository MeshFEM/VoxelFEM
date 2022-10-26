import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIM = 3
# Plot a separate domain to display the BCs
def view_bc(DIM, path, output=None):
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

if __name__ == '__main__':
    view_bc(2, "../../examples/bcs/hump.bc")
