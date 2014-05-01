import numpy as np
import net
from itertools import combinations
import random

trainarray = np.load("/Users/kimducksoo/Desktop/CS51final/vectors.npy")
traingroup = np.load("/Users/kimducksoo/Desktop/CS51final/labels.npy")


(numtrain, dimtrain) = trainarray.shape

trainlabels = [0 for i in range(numtrain)]

for i in range(numtrain):
	if traingroup[i] == 0:
		trainlabels[i] = np.array((1,0))
	else:
		trainlabels[i] = np.array((0,1))

train_data = [0 for i in range(numtrain)]

for j in range(numtrain):
	train_data[j] = (trainarray[j][:]/50,trainlabels[j])

detector = net.Net([dimtrain, 15, 2])
detector.train(train_data, 100, 10, 0.3)

np.save('/Users/kimducksoo/Desktop/CS51final/weights2',detector.weights)
np.save('/Users/kimducksoo/Desktop/CS51final/biases2',detector.biases)
