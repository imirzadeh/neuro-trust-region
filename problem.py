import torch
import numpy as np

class SolverParams():
	# ================= Initial Point =============
	x_0 = np.array([1.0], dtype=np.float32)
	delta_0 = 2.0
	num_samples_interior = 50
	num_samples_boundary  = 4

	# ============= SOLVER (1): INTERPOLATION (neural net) params ======
	neural_net_hiddens = [1, 10, 10, 1]
	interpolation_iterations = 500
	interpolation_learning_rate = 0.05


	#============== SOLVER (2):  DIRECTION params ===========
	beta = 0.8      			# for delta_k    loss
	s1 = 0.1        			# for delta_k    loss
	s2 = 2.0        			# for delta_k    loss

	eta = 0.8 					# for agreement  loss

	zeta_1 = 0.1				# for cauchy     loss
	zeta_2 = 0.1   			    # for cauchy     loss


	gamma_delta = 0.3  	     # for BNTR Loss  
	gamma_agr = 0.3		 # for BNTR Loss  
	gamma_cauchy = 0.3			 # for BNTR Loss  
	direction_iterations = 2501
	direction_learning_rate = 0.01


def f(samples):
	"""
	objective function -> y = x^2
	Implemnted in python and works with numpy
	"""
	result = []
	for s in samples:
		result.append(s[0]**2)
	return result

def f_torch(sample):
	"""
	Objective function implemented in PyTorch
	same as f(samples above), but works with tensors 
	"""
	return torch.pow(sample, 2)