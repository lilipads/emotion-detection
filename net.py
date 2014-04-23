import random

import numpy as np

class Net(object):
	def __init__(self,sizes):
		"""
		Sizes is a list of the number of neurons in each layer of the neural 
		network. For example, if sizes = [3,2,3], that would indicate that the
		neural network has three layers, the first with 3 neurons, the second 
		with 2 neurons, and the third with 3 neurons. 

		The function initializes the parameters and framework for the network.
		"""

		self.numlayers = len(self.sizes)

	def feedforward(self,input):
		"""
		Gives the output of the network given an input. The function feeds
		the input through each layer of the network with the current weights
		and biases.
		"""
		self.sizes = sizes
		self.numlayers = len(sizes)
		self.weights = [np.random.randn((self.sizes[0],self.sizes[1])), 
		  np.random.randn((self.sizes[1],self.sizes[2]))]
		self.biasess = [np.random.randn(self.sizes[1]), np.random.randn(self.sizes[2])]


	def SGD(self, train_data, epochs, mini_batch_size, eta):
		"""
		Trains the network using mini-batch stochastic gradient descent. 
		Parameters:
			train_data: A vector of the data used to train the model. Each element
				in the vector is of the form (x,y) where x is the training data and
				y is the correct output. 

			epochs: The number of epochs in the training.

			mini_batch_size: The number of mini-batches. If we have some set of 
				training data inputs, we partition the set of training data inputs
				into sets of mini_batch_size. The gradient of the cost function
				is calculated for the mini-batch, from which we calculate the 
				gradient for the entire training set and the new weights and biases
				are calculated.

			eta: The constant that determines how much the weights and biases
				change in the direction of the calculated gradient. 
		"""
		for i in range(epochs):
			random.shuffle(train_data) # shuffle the data to randomly devide it into mini batches later
			cur = 0
			while cur < len(train_data):
				mini_batch = train_data[cur : cur + mini_batch_size]
				cur += mini_batch_size
				update_weights(mini_batch, eta)

	def update_weights (self, mini_batch, eta):
		"""
		Updates the weights and biases by applying gradient descent to the 
		mini-batch passed into the function. 
		"""
		delta_weights = [np.zeros((self.sizes[0],self.sizes[1])), np.zeros((self.sizes[1],self.sizes[2]))]
		delta_biases = [np.zeros(self.sizes[1]), np.zeros(self.sizes[2])]
		
		for x, y in mini_batch:
			delta_weights_temp, delta_biases_temp = backprop(x, y)
			for i in range(self.numlayers):
				delta_weights[i] += delta_weights_temp[i]
				delta_biases[i] += delta_biases_temp[i]
		
		# adjust the weights
		for i in range(self.numlayers):
			self.weights[i] += delta_weights[i]
			self.biases[i] += delta_biases[i]

	def backprop(self,x,y):
		"""
		Uses backpropagation to calculate the gradient of the cost function

		Backpropagation consists of the following five steps:
			1) Set the input of the network to the training data. 
			
			2) Use the feedforward function to compute the activations
			at each layer of the network given the input.

			3) Compute each error vector for the last layer before the output
			layer by using the rate of change of the cost as a function of the 
			corresponding output and the derivative of the activation function
			evaluated at the last layer neuron. 

			4) Using each error value in the last layer, calculate the error
			values in each of the previous layers. 

			5) Calculate the gradient of the cost function for each weight and
			for the biases using the error values, using the formula in the spec.
			
		Set activation to a list containing x  # x is the input activation
		Set zlist to empty list  # zlist is the list of z activations
		"""
		sig_prime_vec = np.vectorize(sig_prime)
		sig_vec = np.vectorize(sig)

		activation = []
		activation.append(x)
		zlist = []
		zlist.append(x)

		for i in range(numlayers - 1):
			a = np.dot(self.weights[i],x) + self.biases[i]
			z = sig_vec(a)
			activation.append(a)
			zlist.append(z)

		output = z

		gradcost = a - y  # Gradient of the cost function with respect to the activation outputs

		delt_l = gradcost*sig_prime_vec(output) # Error delta for last layer
		delts = []
		delts.append(delt_l)

		for i in range(2, numlayers+1): # Calculate errors for previous layers
			delt = (np.dot(np.tranpose(weights[numlayers-i]),delts[i])*
					sig_prime_vec(zlist[numlayers-i]))
			delts.append(delt)

		delts = delts.reverse
		grad_biases = delts.pop(0) # Calculate the partial derivatives of the cost function wrt to biases

		grad_weights = []
		for i in range(1, numlayers): # Calculate the gradient with respect to the weights
			gweight = np.dot(delts[i],activation[i-1])
			grad_weights.append(gweight)

		return(grad_weights,grad_biases)

	def evaluate(self,test_data):
		raise TODO


def sig(x):
	return 1.0/(1.0 + np.exp(-x))

def sig_prime(x):
	return sig(x) * (1-sig(x))