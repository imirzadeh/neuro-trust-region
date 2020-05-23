import torch
import numpy as np
from torch.utils import data
from utils import generate_samples
from nn import MLP, train_single_epoch, eval_single_epoch
from problem import SolverParams, f, f_torch

# solver hyper-parameters
params = SolverParams()

def loss_delta(s_k, delta_k, beta, s1, s2):
	"""
	Calculates Delta Loss:
	Delta Loss = -beta x log (s1 + (delta_k - ||s_k||)) + (beta - 1) x ....

	"""
	assert torch.norm(s_k).item() <= delta_k
	return -beta * torch.log(s1 + (delta_k - torch.norm(s_k))) + (beta - 1) * torch.log(s2 - (delta_k - torch.norm(s_k)))

def loss_agreement(x_k, s_k, model, eta):
	"""
	Calculates Agreement Loss:
	Agreement Loss = || ( f(x_k) - f(x_k + s_k)) / ( m(x_k) - m(x_k + s_k)) - eta||
	"""
	agreement = (f_torch(x_k)-f_torch(x_k + s_k))/(model(x_k)-model(x_k + s_k))
	return torch.norm(agreement-eta)

def loss_cauchy(x_k, s_k, delta_k, model, zeta_1, zeta_2):
	"""
	Calculates Cauchy Loss:
	Cauchy Loss = || m(x_k) - m(x_k + s_k) - zeta_1 || grad(x_k) ||x min{grad/hessian, delta_k} ... ||
	"""
	jacobian = torch.norm(torch.autograd.functional.jacobian(model, x_k), p=2)
	hessian = torch.norm(torch.autograd.functional.hessian(model, x_k), p=2)

	cauchy = model(x_k) - model(x_k + s_k) - zeta_1 * jacobian * torch.clamp(jacobian/hessian, max=delta_k)
	return torch.norm(cauchy-zeta_2)

def solver_interpolate(samples, model):
	"""
	First step of solver: interpolation on a list of samples
	"""
	EPOCHS = params.interpolation_iterations

	outputs = f(samples)

	train_x = torch.Tensor(samples)
	train_y = torch.Tensor(outputs)

	loader = data.DataLoader(data.TensorDataset(train_x,train_y))
	criterion = torch.nn.MSELoss() 
	optimizer = torch.optim.SGD(model.parameters(), lr=params.interpolation_learning_rate)

	for train_iter in range(EPOCHS):
		model = train_single_epoch(model, loader, criterion, optimizer)
		if train_iter == EPOCHS-1:
			print("MSE Loss: {0:.4f}".format(eval_single_epoch(model, loader, criterion)['loss']))
	model.eval()
	return model


def solver_direction_and_step(x_k, delta_k, model):
	"""
	Second step of solver: finding the argmin of model to be the new x_k
	"""
	x_k = torch.from_numpy(x_k)
	s_k = torch.randn(x_k.shape, requires_grad=True)

	optimizer = torch.optim.SGD([s_k], lr=params.direction_learning_rate)

	for iteration in range(params.direction_iterations):
		optimizer.zero_grad()

		L_delta = loss_delta(s_k, delta_k, params.beta, params.s1, params.s2)
		L_agr = loss_agreement(x_k, s_k, model, eta=params.eta)
		L_cauchy = loss_cauchy(x_k, s_k, delta_k, model, params.zeta_1, params.zeta_2)


		# calculate total loss 
		loss = params.gamma_delta*L_delta + params.gamma_agr * L_agr + params.gamma_cauchy*L_cauchy
		if iteration % 500 == 0:
			# print(L_delta.item(), L_agr.item(), L_cauchy.item())
			# print(s_k.item(), ' => ', loss.item())
			print('iteratiom: {:04} | s_k: {:.3f}, loss_total: {:.3f} (delta: {:.3f}, agr: {:.3f}, cauchy: {:.3f})'.format(iteration, s_k.item(), loss.item(), L_delta.item(), L_agr.item(), L_cauchy.item()))
		loss.backward()
		optimizer.step()

	new_x_k = x_k + s_k
	print("New x_k is ", new_x_k.item())
	return new_x_k




def solve_one_iteration(iteration=1):
	x_k = params.x_0
	delta_k = params.delta_0

	samples = generate_samples(params.num_samples_interior, params.num_samples_boundary, x_k, delta_k)
	
	print("----------------- iteration {} -----------------".format(iteration))
	print('******* 1. Training the neural network with MSE Loss ******')

	model = MLP(layers=[1, 10, 10, 1])
	model = solver_interpolate(samples, model)

	print('******* 2. Calculating argmin of the neural network ******')
	solver_direction_and_step(x_k, delta_k, model)


if __name__ == "__main__":
	solve_one_iteration()

