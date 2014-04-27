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

        self.sizes = sizes
        self.numlayers = len(sizes)
        self.weights = [np.random.randn(sizes[i],sizes[i+1]) for i in range(self.numlayers - 1)]
        self.biases = [np.random.randn(sizes[i + 1]) for i in range(self.numlayers - 1)] 

    def feedforward(self,a):
        for i in range(self.numlayers - 1):
            z = np.dot(a,self.weights[i]) + self.biases[i]
            a = sig_vec(z)
        return a

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
            
            j = 0
            while cur < len(train_data):
                j += 1
                mini_batch = train_data[cur : cur + mini_batch_size]
                cur += mini_batch_size
                self.update_weights(mini_batch, eta)

    def update_weights (self, mini_batch, eta):
        """
        Updates the weights and biases by applying gradient descent to the 
        mini-batch passed into the function. 
        """
        delta_weights = [np.zeros((self.sizes[i],self.sizes[i+1])) for i in range(self.numlayers - 1)]
        delta_biases = [np.zeros(self.sizes[i + 1]) for i in range(self.numlayers - 1)] 
        
        for x, y in mini_batch:
            delta_weights_temp, delta_biases_temp = self.backprop(x, y)
            for i in range(self.numlayers-1):
                delta_weights[i] += delta_weights_temp[i]
                delta_biases[i] += delta_biases_temp[i]
        
        # adjust the weights
        for i in range(self.numlayers-1):
            self.weights[i] += - eta * delta_weights[i]
            self.biases[i] += - eta * delta_biases[i]
        
        """
        for x, y in mini_batch:
            print self.evaluate(x),
        print "after: biases", self.biases[0][300]
        """

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
        numlayers = self.numlayers
        sizes = self.sizes
        
        zlist = [np.zeros(sizes[i]) for i in range(self.numlayers)] # weighted input before applying sigmoid function
        zlist[0] = x # weighted input in the first layer is just the input
        activation = [np.zeros(sizes[i]) for i in range(self.numlayers)] # output after applying sigmoid function
        activation[0] = x
        
        # feedforward: compute the output in each layer
        for i in range(1, numlayers):
            zlist[i] = np.dot(activation[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            activation[i] = sig_vec(zlist[i])
            
        gradcost = activation[numlayers - 1] - y  # Gradient of the cost function with respect to the activation outputs 
        
        delts = [0 for i in range(numlayers)]
        delts[self.numlayers - 1] = gradcost * sig_prime_vec(zlist[numlayers - 1]) # last layer
        
        # Backward propagate: calculate errors for previous layers
        for i in range(numlayers - 2, -1, -1): 
            delts[i] = (np.dot(delts[i + 1], np.transpose(self.weights[i])) * sig_prime_vec(zlist[i]))
        
        # calculate the gradients
        grad_weights = [np.zeros((sizes[i],sizes[i+1])) for i in range(self.numlayers - 1)]
        for i in range(1, numlayers): # Calculate the partial derivatives of the cost function wrt the weights
            grad_weights[i - 1] = np.dot(np.transpose(activation[i - 1].reshape((1,-1))), delts[i].reshape((1,-1))) # check later

        grad_biases = delts[1:] # Calculate the partial derivatives of the cost function wrt biases

        return(grad_weights,grad_biases)

    def evaluate(self,test_data):
        """
        sig_prime_vec = np.vectorize(sig_prime)
        sig_vec = np.vectorize(sig)

        activation = test_data
        
        for i in range(1, self.numlayers):
            z = np.dot(activation, self.weights[i - 1]) + self.biases[i - 1]
            activation = sig_vec(z)

        # return activation
        return max(enumerate(activation),key=lambda x: x[1])[0] # return the index of the maximum element in output list
        """
        for (x,y) in test_data:
            output = [(np.argmax(self.feedforward(x)), np.argmax(y))]
        return sum(int(x == y) for (x, y) in output)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def sig_prime(x):
    return sig(x) * (1-sig(x))

sig_prime_vec = np.vectorize(sig_prime)
sig_vec = np.vectorize(sig)
        