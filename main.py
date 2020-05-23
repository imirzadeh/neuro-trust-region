import torch
import numpy as np
from torch.utils import data
from utils import generate_samples
from nn import MLP, train_single_epoch, eval_single_epoch


# objective function -> y = x^2
def objective(samples):
	result = []
	for s in samples:
		result.append(s[0]**2)
	return result

def f_torch(sample):
	return torch.pow(sample, 2)

def solver_interpolate(samples, model):
	learning_rate = 0.01
	EPOCHS = 500

	outputs = objective(samples)

	train_x = torch.Tensor(samples)
	train_y = torch.Tensor(outputs)

	loader = data.DataLoader(data.TensorDataset(train_x,train_y))
	criterion = torch.nn.MSELoss() 
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	for train_iter in range(EPOCHS):
		model = train_single_epoch(model, loader, criterion, optimizer)
		if train_iter == EPOCHS-1:
			print(eval_single_epoch(model, loader, criterion))
	model.eval()
	return model


def loss_delta(s_k, beta, s1, s2, delta_k):
	return -beta * torch.log(s1 + (delta_k - torch.norm(s_k))) + (beta - 1) * torch.log(s2 - (delta_k - torch.norm(s_k)))

def loss_agreement(x_k, s_k, model, eta):
	agreement = (f_torch(x_k)-f_torch(x_k + s_k))/(model(x_k)-model(x_k + s_k))
	return torch.norm(agreement-eta)

def loss_cauchy(x_k, s_k, delta_k, model, alpha_1, alpha_2):
	jacobian = torch.norm(torch.autograd.functional.jacobian(model, x_k), p=2)
	hessian = torch.norm(torch.autograd.functional.hessian(model, x_k), p=2)
	cauchy = model(x_k) - model(x_k + s_k) - alpha_1 * jacobian * torch.clamp(jacobian/hessian, max=delta_k)
	return torch.norm(cauchy-alpha_2)

def solver_direction_and_step(x_k, delta_k, model):
	beta = 0.5
	s1 = 0.1
	s2 = delta_k
	eta = 0.8
	alpha_1 = 0.1
	alpha_2 = 0.1
	x_k = torch.from_numpy(x_k)
	s_k = torch.randn(x_k.shape, requires_grad=True)

	optimizer = torch.optim.SGD([s_k], lr=0.01)

	for iteration in range(5001):
		L_delta = loss_delta(s_k, beta, s1, s2, delta_k)
		L_agr = loss_agreement(x_k, s_k, model, eta=eta)
		L_cauchy = loss_cauchy(x_k, s_k, delta_k, model, alpha_1=alpha_1, alpha_2=alpha_2)
		

		optimizer.zero_grad()
		loss = 0.0*L_delta + 0.1 * L_agr + 0.45*L_cauchy
		if iteration % 500 == 0:
			print(s_k.item()	, ' => ', loss.item())
		loss.backward()
		optimizer.step()

	new_x_k = x_k + s_k
	print("New x_k is ", new_x_k.item())




def solve_one_iteration(iteration=1):
	x_k = np.array([1.0], dtype=np.float32)
	delta_k = 2
	num_samples_interior = 50
	num_samples_boundary  = 4
	samples = generate_samples(num_samples_interior, num_samples_boundary, x_k, delta_k)
	
	print("------------ iteration {} -------------")
	print('Training the neural network with MSE Loss')

	model = MLP(layers=[1, 10, 10, 1])
	model = solver_interpolate(samples, model)

	print('calculating other losses')
	solver_direction_and_step(x_k, delta_k, model)


if __name__ == "__main__":
	# x_k = np.array([1.0, 1.0, 1.0])
	# num_samples = 4
	# delta = 1.0
	# samples = generate_samples(num_samples, x_k, delta)
	# print(samples)
	# print()
	# print(objective(samples))
	test_tr()