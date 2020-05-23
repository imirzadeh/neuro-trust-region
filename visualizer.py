import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import torch
from nn import MLP
from problem import SolverParams

params = SolverParams()

if __name__ == "__main__":

    model = MLP(params.neural_net_hiddens)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(params.x_0[0]-params.delta_0, params.x_0[0]+params.delta_0, 0.05)
    Y = np.arange(params.x_0[1]-params.delta_0, params.x_0[0]+params.delta_0, 0.05)
    X, Y = np.meshgrid(X, Y)

    Z=[]
    for j in range(len(X)):
        for i in range(len(X[0])):
        # your loop order was backwards
            point = np.array([[X[j][i],Y[j][i]]], dtype=np.float32)
            inp = torch.from_numpy(point)
            Z.append(model(inp).data.numpy()/1000)
    
    Z = np.array(Z).reshape(X.shape)

    # R = np.sqrt(X**2 + Y**2)
    # print(R.shape)
    # Z = np.sin(R)


    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 10.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel(r'$f(x, y) \times 10^{-3}$')


    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('result.png', dpi=200)