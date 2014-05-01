import numpy as np
import net

trainarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/mtraindata.csv", dtype=float, delimiter=',')
testarray = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/mtestdata.csv", dtype=float, delimiter=',')
traingroup = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/mtraingroup.csv", dtype=float, delimiter=',') 
testgroup = np.genfromtxt("/Users/kimducksoo/Desktop/CS51final/mtestgroup.csv", dtype=float, delimiter=',') 

testarray = np.array((testarray[0],testarray[1],testarray[2],testarray[3],testarray[4],trainarray[54],trainarray[56],testarray[5], testarray[6],testarray[7],testarray[10],trainarray[61],trainarray[62],trainarray[66],testarray[14],trainarray[90],trainarray[91],trainarray[92],trainarray[103],testarray[19]))
trainarray = np.delete(trainarray,np.array((2,9,10,14,24,32,41,42,53,62,63,67,70,72,74,76,79,81,82,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104)) - 1,0)
traingroup = np.delete(traingroup,np.array((2,9,10,14,24,32,41,42,53,62,63,67,70,72,74,76,79,81,82,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104)) - 1,0)
testgroup = np.array((0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1))

(numtrain, dimtrain) = trainarray.shape
(numtest, dimtest) = testarray.shape

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
	test_data[j] = (testarray[j][:]/50,testlabels[j])

detector = net.Net([dimtrain, 20, 2])
detector.train(train_data, 100, 10, 0.3)
numcorrect = detector.evaluate(test_data)
print "Correct: ", numcorrect

