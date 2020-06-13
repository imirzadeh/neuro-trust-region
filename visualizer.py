import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import torch
from nn import MLP
from problem import SolverParams, f
from pathlib import Path

Path("./results").mkdir(parents=True, exist_ok=True)
params = SolverParams()

def get_evals(approx):
    if approx == True:
        model = MLP(params.neural_net_hiddens)
        model.load_state_dict(torch.load('./model.pth'))
        model.eval()
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
            if approx == False:
                val = f(point)
            else:
                val = model(inp).data.numpy()    
            Z.append(val)
    
    Z = np.array(Z).reshape(X.shape)
    return X, Y, Z

def plot_comparison():

    
    #################### REAL ###################
    approx = False
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    X, Y, Z = get_evals(approx=False)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel(r'$f(x, y)$')
    title = 'approximation' if approx else 'real'
    ax.set_title('function surface ({})'.format(title))



    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    ax = fig.add_subplot(2, 2, 2)
    cp = ax.contour(X, Y, Z, levels=20)
    ax.clabel(cp, inline=True, 
              fontsize=10)

    title = 'approximation' if approx else 'real'
    ax.set_title('Contour Plot ({})'.format(title))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    ################ Approximation ###################

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    approx = True
    X, Y, Z = get_evals(approx=True)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel(r'$f(x, y)$')
    title = 'approximation' if approx else 'real'
    ax.set_title('function surface ({})'.format(title))


    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax = fig.add_subplot(2, 2, 4)
    cp = ax.contour(X, Y, Z, levels=20)
    ax.clabel(cp, inline=True, 
              fontsize=10)

    title = 'approximation' if approx else 'real'
    ax.set_title('Contour Plot ({})'.format(title))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')


    plt.savefig('./results/comparison.png', dpi=200)

def plot_trajectory(history_x):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    X, Y, Z = get_evals(approx=False)
    print("shapes >>> ", X.shape, Y.shape, Z.shape)
    cp = ax.contour(X, Y, Z, levels=20)
    ax.clabel(cp, inline=True, 
              fontsize=10)

    x  = [x[0] for x in history_x]
    y  = [x[1] for x in history_x]

    ax.plot(x, y, marker='o', color='red')
    for i, x in enumerate(history_x):
        plt.text(x[0]+0.02, x[1]+0.02, r'$x_{}$'.format(i), fontsize=15)
    ax.set_title("Trajectory")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    plt.savefig('./results/trajectory.png', dpi=200)


if __name__ == "__main__":
    # plot_comparison()
    plot_trajectory([[0.0, 0.0], [0.25, 0.25], [0.7, 0.7]])