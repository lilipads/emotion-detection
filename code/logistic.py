import numpy as np
from PIL import Image

class Logistic(object):
    """
    randomly initialize weights
    """
    def __init__(self, dim):
        self.dim = dim
        self.weights = np.random.normal(0, 1, (1, dim)) 

    """
    evaluate the probability of belonging to class 1
    P(C|phi_n) = sigmoid(w.T * phi_n)
    """
    def evaluate(self,phi_n):
        return sig(np.dot(self.weights, phi_n.T))

    """
    update the weights using gradient descent until convergence
    phi is a matrix of feature vectors, one row for each datum
    labels is a 1 X N array of binary labels
    """
    def train(self, phi, labels, max_iter = 1000, learn_rate = 0.01):
        N = len(labels) # number of training data
        dim = self.dim # dimension of feature space
        it = 0
        while True:
            it += 1
            if it > max_iter:
                break
            
            # gradient of error function: grad_E = sum{(y - t) * phi}
            grad_E = [0 for i in range(dim)]
            for n in xrange(N):
                y_n = self.evaluate(phi[n]) # predicted probability
                grad_E += (y_n - labels[n]) * phi[n] 

            w_prev = self.weights
            # update weights using gradient descent
            self.weights = w_prev - (learn_rate * np.array(grad_E)) 
            converged = False
            for x in (self.weights - w_prev):
                for y in x:
                    if abs(y) < 0.0001:
                        converged = True
                        break
            if converged:
                print 'Gradient descent converged in ' + str(it) + ' iterations'
                print 'LOGISTIC REGRESSION training complete!' 
                break

    """
    predict test data: return 1 if the datum is more probable to be in class 1 
    and 0 if the datum is more probable to be in class 0
    round to the closest integer
    """
    def predict(self,phi_n):
        return int(round(self.evaluate(phi_n)[0]))

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))