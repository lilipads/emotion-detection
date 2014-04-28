import numpy as np
import net

trainarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/traindata.csv", dtype=float, delimiter=',')
testarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/testdata.csv", dtype=float, delimiter=',')
traingroup = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/groupdata.csv", dtype=float, delimiter=',') - 1

(numtrain, dimtrain) = trainarray.shape
(numtest, dimtest) = testarray.shape

trainlabels = [0 for i in range(numtrain)]
testlabels = [0 for i in range(numtest)]

for i in range(numtrain):
	if traingroup[i] == 0:
		trainlabels[i] = np.array((1,0))
	else:
		trainlabels[i] = np.array((0,1))

print trainlabels

testgroup = np.array((1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2)) - 1
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
	test_data[j] = (testarray[j][:]/50,testlabels[j])

for i in range(5, 200, 5):
	detector = net.Net([dimtrain, i, 2])
	detector.SGD(train_data, 50, 10, 0.3)
	numcorrect = detector.evaluate(test_data)
	print "N: ", i, "; Correct: ", numcorrect