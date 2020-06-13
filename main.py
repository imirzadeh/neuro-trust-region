import torch
import numpy as np
from torch.utils import data
import torch_optimizer as toptim
from utils import generate_samples, normalize
from problem import SolverParams, f, f_torch
from nn import MLP, train_single_epoch, eval_single_epoch, eval_single_epoch_debug

from visualizer import plot_trajectory, plot_comparison
# solver hyper-parameters
params = SolverParams()

def loss_delta(s_k, delta_k):
    """
    Calculates Delta Loss:
    Delta Loss = -beta x log (s1 + (delta_k - ||s_k||)) + (beta - 1) x ....

    """
    if torch.norm(s_k) <= delta_k:
        return torch.Tensor([0.0])
    else:
        norm_s_k = torch.norm(s_k, p=2)
        return (norm_s_k * torch.exp(norm_s_k-delta_k)) - norm_s_k**2/2.0 + delta_k**2/2.0
    

def loss_agreement(x_k, s_k, model, eta):
    """
    Calculates Agreement Loss:
    Agreement Loss = || ( f(x_k) - f(x_k + s_k)) / ( m(x_k) - m(x_k + s_k)) - eta||
    """
    rho = (f_torch(x_k)-f_torch(x_k + s_k))/(model(x_k)-model(x_k + s_k))
    return torch.cosh(rho-eta)
    # return torch.norm(agreement-eta) # the old agreement loss 


def loss_cauchy(x_k, s_k, delta_k, model, zeta_1, zeta_2):
    """
    Calculates Cauchy Loss:
    Cauchy Loss = || m(x_k) - m(x_k + s_k) - zeta_1 || grad(x_k) ||x min{grad/hessian, delta_k} ... ||
    """

    # # the old implementation was:
    # jacobian = torch.norm(torch.autograd.functional.jacobian(model, x_k), p=2)
    # hessian = torch.norm(torch.autograd.functional.hessian(model, x_k), p=2)
    # cauchy = model(x_k) - model(x_k + s_k) - zeta_1 * jacobian * torch.clamp(jacobian/hessian, max=delta_k)
    # return torch.norm(cauchy-zeta_2)
    return torch.norm(model(x_k+s_k)-model(x_k) + zeta_1 * torch.norm(s_k , p=2)**2 - zeta_2, p=2)


def loss_local(x_k, s_k, gamma_2_prime, c, gamma_2_double_prime):
    """
    Calculates the Local Loss
    """
    jacobian_norm = torch.mean(torch.norm(torch.autograd.functional.jacobian(model, x_k + s_k), p=2))
    hessian = torch.autograd.functional.hessian(model, x_k)
    smallest_eigenval, _ = torch.symeig(hessian)
    smallest_eigenval = torch.min(smallest_eigenval)
    # print("jac", jacobian_norm.item(), "lambda", smallest_eigenval.item())
    local_loss = gamma_2_prime * torch.norm(smallest_eigenval-c, 2) + gamma_2_double_prime * jacobian_norm / x_k.shape[0]
    return local_loss


def solver_interpolate(samples, model):
    """
    First step of solver: interpolation on a list of samples
    """
    EPOCHS = params.interpolation_iterations

    outputs = f(samples)

    train_x = torch.Tensor(samples)
    train_y = torch.Tensor(outputs)
    loader = data.DataLoader(data.TensorDataset(train_x,train_y), batch_size=128)

    samples_test = np.array([[-1.5, 1.0], [-1.2, 1.0], [-0.7, 1.0], [0, 1.]], dtype=np.float32)
    test_x = torch.Tensor(samples_test)
    test_y = torch.Tensor(f(samples_test))
    loader_test = data.DataLoader(data.TensorDataset(test_x,test_y), batch_size=8)

    criterion = torch.nn.MSELoss() 
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.interpolation_learning_rate, momentum=0.8)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=params.interpolation_learning_rate, max_iter=25, line_search_fn='strong_wolfe')
    optimizer = toptim.RAdam(model.parameters(), lr=params.interpolation_learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    
    for train_iter in range(EPOCHS):
        model = train_single_epoch(model, loader, criterion, optimizer)
        if train_iter == EPOCHS-1:
            print("Finished interpolation phase")
            print("iter: {}".format(train_iter))
            print("MSE Loss: {0:.4f}".format(eval_single_epoch(model, loader, criterion)['loss']))
            eval_single_epoch_debug(model, loader_test, criterion)
    torch.save(model.state_dict(), './model.pth')

    model.eval()
    return model


def solver_direction_and_step(x_k, delta_k, model):
    """
    Second step of solver: finding the argmin of model to be the new x_k
    """
    x_k = torch.from_numpy(x_k)
    s_k = torch.rand(x_k.shape, requires_grad=True)

    # optimizer = toptim.RAdam([s_k], lr= 1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
    optimizer = torch.optim.SGD([s_k], lr=params.direction_learning_rate, momentum=0.5)

    for iteration in range(params.direction_iterations):
        optimizer.zero_grad()

        L_delta = loss_delta(s_k, delta_k)
        L_agr = loss_agreement(x_k, s_k, model, params.eta)
        L_cauchy = loss_cauchy(x_k, s_k, delta_k, model, params.zeta_1, params.zeta_2)
        L_local = loss_local(x_k, s_k, params.gamma_2_prime, params.c, params.gamma_2_double_prime)

        # calculate total loss 
        #loss = params.gamma_delta*L_delta + params.gamma_agr * L_agr + params.gamma_cauchy*L_cauchy
        loss = params.gamma_agr * L_agr + params.gamma_cauchy*L_cauchy + params.gamma_local * L_local
        if iteration % 500 == 0:
            print('iteratiom: {:04} | s_k: {}, loss_total: {:.3f} (delta: {:.3f}, agr: {:.3f}, cauchy: {:.3f}), local: {:.3f}'\
                .format(iteration, s_k.data, loss.item(), L_delta.item(), L_agr.item(), L_cauchy.item(), L_local.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([s_k], 0.5)
        optimizer.step()

    s_k_final = s_k.detach().numpy()
    if np.linalg.norm(s_k_final, ord=2) > delta_k:
        new_x_k = x_k + torch.from_numpy(delta_k*normalize(s_k_final))
    else:
        new_x_k = x_k + s_k
    return new_x_k.detach().numpy()



def solve_one_iteration(x_k, delta_k, model, iteration=1):

    samples = generate_samples(params.num_samples_interior, params.num_samples_boundary, x_k, delta_k)
    
    print("----------------------------------- iteration {} ----------------------------".format(iteration))
    print('******* 1. Training the neural network with MSE Loss for interpolation ******')
    model = solver_interpolate(samples, model)

    print('********** 2. Calculating  the best direction in the region ********')
    new_x_k = solver_direction_and_step(x_k, delta_k, model)
    print("new_x_k: ", new_x_k)
    return new_x_k, model


if __name__ == "__main__":
    # initial values
    x_k = params.x_0
    delta_k = params.delta_0
    model = MLP(layers=params.neural_net_hiddens)
    
    history = [x_k]

    for iteration in [1, 2, 3, 4, 5, 6]:
        new_x_k, model = solve_one_iteration(x_k, delta_k, model, iteration)
        x_k = new_x_k
        history.append(x_k)

    print("************** trajectory *************")
    for i, x in enumerate(history):
        print("{} => x_k={}, f(x_k)={}".format(i+1, list(x), list(f([x]))))
    # plot_trajectory(history)
    # plot_comparison()


