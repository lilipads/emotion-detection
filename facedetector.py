import numpy as np
import facenet

trainarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/traindata.csv", dtype=float, delimiter=',')
testarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/testdata.csv", dtype=float, delimiter=',')
traingroup = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/groupdata.csv", dtype=float, delimiter=',')
traingroup = traingroup - 1
testgroup = np.array((1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2)) - 1

(numtrain, dimtrain) = trainarray.shape
(numtest, dimtest) = testarray.shape

train_data = [0 for i in range(numtrain)]
test_data = [0 for i in range(numtest)]

for j in range(numtrain):
	train_data[j] = (trainarray[j][:]/50,traingroup[j])

for j in range(numtest):
	test_data[j] = (testarray[j][:]/50,testgroup[j])

for i in range(5, 200, 5):
	for j in [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
		detector = facenet.Net([dimtrain, i, 1])
		detector.train(train_data, 50, 10, j)
		numcorrect = detector.evaluate(test_data)
		print "N: ", i, "; Correct: ", numcorrect