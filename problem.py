import torch
import numpy as np

def bowlـ2d_python(samples):
    """
    objective function -> y = x^2
    Implemnted in python and works with numpy
    """
    result = []
    for s in samples:
        result.append([s[0]**2 + s[1]**2])
    return result

def bowl_2d_torch(sample):
    """
    Objective function implemented in PyTorch
    same as f(samples above), but works with tensors 
    """
    return torch.norm(sample, p=2)


def rosenbrock_2d_torch(sample):
    """
    2D rosenbrock function: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html
    """
    x, y = sample
    return ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)

def rosenbrock_2d_python(samples):
    """
    2D rosenbrock function: http://benchmarkfcns.xyz/benchmarkfcns/rosenbrockfcn.html
    """
    result = []
    for s in samples:
        x, y = s[0], s[1]
        ans = ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2 )
        result.append([ans])
    return result
    #real_jac = torch.norm(torch.Tensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2))), p=2)

f = rosenbrock_2d_python
f_torch = rosenbrock_2d_torch

# f = bowlـ2d_python
# f_torch = bowl_2d_torch


class SolverParams():
    # ================= Initial Point =============
    x_0 = np.array([0.5, 0.5], dtype=np.float32)    
    delta_0 = 0.5
    num_samples_interior = 82
    num_samples_boundary  = 18

    # ============= SOLVER (1): INTERPOLATION (neural net) params ======
    neural_net_hiddens = [2, 25, 25, 1]
    interpolation_iterations = 15000
    interpolation_learning_rate = 0.001
    stability_scale = 1

    #============== SOLVER (2):  DIRECTION params ===========
    beta = 0.5                  # for delta_k    loss
    s1 = 0.02                   # for delta_k    loss
    s2 = 2.0                    # for delta_k    loss

    eta = 0.8                    # for agreement  loss

    zeta_1 = 1.0                 # for cauchy     loss
    zeta_2 = 0                   # for cauchy     loss


    gamma_delta = 0.0        # for BNTR Loss  
    gamma_agr = 0.5      # for BNTR Loss  
    gamma_cauchy = 0.5           # for BNTR Loss  
    direction_iterations = 5001
    direction_learning_rate = 0.01

if __name__ == "__main__":
    print(rosenbrock_2d_python([[0.0, 0.0], [0.5, 0.5]]))
