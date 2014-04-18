"""Andrew"""

class Layer(object):
	"""
	A representation of one layer of neurons in the network.
	Each layer has 'cn' neurons, 'ci' inputs, and 'co' outputs. 
	"""

	def __init__(self, ci, cn, co):
	"""
	Initializes the layer object. The initialization function takes
	in three parameters:
		ci: number of inputs
		cn: number of neurons
		co: number of outputs 
	"""
		pass

	def step(self, inp):
	"""
	Performs one step of the layer. Given an input 'inp', calculate the 
	output of the layer and update the object layer's input and output. 
	"""
		pass

	def init(self):
	"""
	Initializes the layer with random values.
	"""
		pass 

class Network(object):
	"""
	A class for the structure of the neural network as a whole.  
	"""
	def __init__(self, co, layers, trainf, errorf):
	"""
	Initializes the network. Takes the following paramters:
		co: int
			Number of outputs
		layers: Layer list
			List of the layers in the network
		trainf: callable
			Function that takes the network, a set of inputs, and the target
			categories for each input. The function runs the inputs through
			the network and compares the outputs to the targets. The function
			then backpropagates the errors and adjusts the weights accordingly.
			The function returns the network with new weights. 
		errorf: callable
			Function that calculates the sum-of-squares error given the 
			current neural network, a set of inputs, and the corresponding
			targets. 
	"""
		pass

	def step(self, inp):
	"""
	Runs through the neural network given an input vector. Returns the output
	after running the input through each layer of the neural network.
	"""
		pass

	def init(self):
	"""
	Initializes the neural network by running through each layer and
	calling the 'init' function from each layer object. The neural
	network is initialized with random values.
	"""
		pass

	def train(self, *args, **kwargs):
	"""
	Trains the neural network given input and target vectors. 
	See below for further details about training. 
	"""

	def reset(self):
	"""
	Set each input and output value in the network to 0
	"""

class TanSig:
    """
    Hyperbolic tangent sigmoid transfer function

    Takes an array of input values and returns an array of the
    corresponding hyperbolic tangent functions.
    """

""" Lili"""
class Trainer(object):
	def __init__(self, Train, epochs=500, goal=0.01, show=100, **kwargs):

	def __str__(self):

	def __call__(self, net, input, target=None, **kwargs):

class Train(object):
	def __init__(self, epochf, epochs):

	def __call__(self, net, *args):

	def error(self, net, input, target, output=None):

class TrainGD(Train):
	def __init__(self, net, input, target, lr=0.01, adapt=False):

	def __call__(self, net, input, target):

	def calc(self, net, input, target):

	def learn(self, net, grad):

class SSE():
    def __call__(self, e):

    def deriv(self, e):

def newnet (minmax, size, transf=None)
	""" Create a network that will """
