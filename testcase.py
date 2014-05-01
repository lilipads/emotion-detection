import numpy as np
import net
from itertools import combinations
import random

trainarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/traindata3.csv", dtype=float, delimiter=',')
testarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/testdata3.csv", dtype=float, delimiter=',')
traingroup = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/traingroup3.csv", dtype=float, delimiter=',') 
testgroup = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/testgroup3.csv", dtype=float, delimiter=',') 
realtestarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/testcase.csv", dtype=float, delimiter=',')

neutralarray = np.concatenate((trainarray[0:45],testarray[0:10]))
happyarray = np.concatenate((trainarray[45:61],testarray[10:16]))

for i in range(10):

	trainarray = np.concatenate((neutralarray,happyarray))
	testarray = realtestarray
	traingroup = [0 for i in range(55)] + [1 for i in range(22)]
	testgroup = [1]

	(numtrain, dimtrain) = trainarray.shape
	numtest = 1

	trainlabels = [0 for i in range(numtrain)]
	testlabels = [0 for i in range(numtest)]


	for i in range(numtrain):
		if traingroup[i] == 0:
			trainlabels[i] = np.array((1,0))
		else:
			trainlabels[i] = np.array((0,1))

	for i in range(numtest):
		if testgroup[i] == 0:
			testlabels[i] = np.array((1,0))
		else:
			testlabels[i] = np.array((0,1))

	train_data = [0 for i in range(numtrain)]
	test_data = [0 for i in range(numtest)]

	for j in range(numtrain):
		train_data[j] = (trainarray[j][:]/50,trainlabels[j])

	for j in range(numtest):
		test_data[j] = (testarray/50,testlabels[j])

	detector = net.Net([dimtrain, 15, 2])
	detector.SGD(train_data, 100, 10, 0.3)
	numcorrect = detector.evaluate(test_data)
	print "Correct: ", numcorrect

