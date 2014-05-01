import numpy as np
import net
from itertools import combinations
import random
from vectorize import vectorize

weights = np.load('/Users/kimducksoo/Desktop/CS51final/weights3.npy')
biases = np.load('/Users/kimducksoo/Desktop/CS51final/biases3.npy')
testarray = np.load('/Users/kimducksoo/Desktop/CS51final/camtestarray.npy')
testgroup = [1,0,1,0,0,1]

numtest = 6

testlabels = [0 for i in range(numtest)]

for i in range(numtest):
	if testgroup[i] == 0:
		testlabels[i] = np.array((1,0))
	else:
		testlabels[i] = np.array((0,1))

test_data = [0 for i in range(numtest)]

for j in range(numtest):
	test_data[j] = (testarray[j][:]/134 ,testlabels[j])

detector = net.Net([280, 2])
detector.weights = weights	
detector.biases = biases
numcorrect = detector.evaluate(test_data)
print "Correct: ", numcorrect

