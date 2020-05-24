import torch
import torch.nn as nn
from problem import SolverParams

CLIP_VAL = 0.5#0.25
SCALE_N = SolverParams().stability_scale

class MLP(nn.Module):
	"""
	Multi Layer Perceptron for approximation 
	"""
	def __init__(self, layers):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(layers[0], layers[1])
		self.W2 = nn.Linear(layers[1], layers[2])
		self.W3 = nn.Linear(layers[2], layers[3])


	def forward(self, x):
		out = torch.sigmoid(self.W1(x))
		out = torch.sigmoid(self.W2(out))
		out = self.W3(out) * SCALE_N
		return out



def train_single_epoch(net, loader, criterion, optimizer):
	"""
	Training the neural network for singlee epoch
	"""
	net.train()	
	
	for batch_idx, (data, target) in enumerate(loader):
		def closure():
			optimizer.zero_grad()
			pred = net(data)
			loss = criterion(pred, target)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_VAL)
			return loss
		optimizer.step(closure)
	return net

def eval_single_epoch(net, loader, criterion):
	"""
	Evaluating the neural network for single epoch
	"""
	net.eval()
	test_loss = 0
	with torch.no_grad():
		for data, target in loader:
			output = net(data)
			test_loss += criterion(output, target).item()
	test_loss /= len(loader.dataset)
	return {'loss': test_loss}

def eval_single_epoch_debug(net, loader, criterion):
	"""
	Evaluating the neural network for single epoch
	"""
	net.eval()
	test_loss = 0
	with torch.no_grad():
		for data, target in loader:
			output = net(data)
			print(data)
			print(output.data)
			print(target.data)
			print("**"*10)
			test_loss += criterion(output, target).item()
	test_loss /= len(loader.dataset)
	return {'loss': test_loss}
