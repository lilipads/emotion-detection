class Layer(object):
	def __init__(self, ci, cn, co, property):

	def step(self, inp):

	def init(self):

class Perceptron(Layer):
	def __init__(self, ci, cn, transf):

	def _step(self, inp):

class Network(object):
	def __init__(self, inp_minmax, co, layers, connect, trainf, errorf):
		pass

	def step(self, inp):
		pass

	def sim(self, input):

	def init(self):

	def train(self, *args, **kwargs):

	def reset(self):

	def save(self, fname):

	def copy(self):

class TanSig:
    """
    Hyperbolic tangent sigmoid transfer function
    """

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

class MSE():
    def __call__(self, e):

    def deriv(self, e):

def newnet (minmax, size, transf=None)
	""" Create a network that will """
