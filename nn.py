import torch
import torch.nn as nn
import torch.nn.functional as F

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
		return self.W3(out)



def train_single_epoch(net, loader, criterion, optimizer):
	"""
	Training the neural network for singlee epoch
	"""
	net.train()	
	
	for batch_idx, (data, target) in enumerate(loader):
		optimizer.zero_grad()
		pred = net(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
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
