import sys
from math import exp
from math import log
from math import sqrt
import math
import re
import operator
from random import randrange
import random
import copy
import matplotlib.pyplot as plt
import operator

l = float(sys.argv[1])
h = int(sys.argv[2])
e = int(sys.argv[3])
trainFile = sys.argv[4]
testFile = sys.argv[5]
threshold = 0.5

def preProcessingData(fileName):
	readFile = open(fileName, 'r')
	data = readFile.readlines()
	dataTmp = []
	labels = []
	labelTypeList = []
	dataSet = []
	labelType = []
	isData = 0 #label for judging if the next line is data
	for line in data:
		line = line.strip('\r').strip('\n').rstrip(' ').lstrip(' ')
		dataTmp = line.split(None, 2)
		if len(dataTmp) == 0:
			continue
		if len(re.findall('@attribute', dataTmp[0], re.I)):
			labels.append(dataTmp[1].strip('\'').strip('\"'))
			labelType = dataTmp[2].strip('{').strip('}').replace(' ', '').replace('\'', '').replace('\"', '').split(',')
			labelTypeList.append(labelType)
		elif len(re.findall('@data', dataTmp[0], re.I)):
			isData = 1
			continue
		if isData == 1:
			dataTmp = line.replace(' ', '').replace('\'', '').replace('\"', '').split(',')
			for n in range(len(dataTmp)):
				if len(labelTypeList[n]) == 1:
					dataTmp[n] = float(dataTmp[n])
			dataSet.append(dataTmp)
	return labels, dataSet, labelTypeList

def sigmoid(x):
	if isinstance(x, list):
		res = []
		for e in x:
			res.append(1 / (1 + exp(-e)))
		return res
	else:
		return 1 / (1 + exp(-x))

def getOneOfk(instance, labelTypeList):
	res = []
	for n in range(len(instance)):
		element = instance[n]
		if isinstance(element, str):
			tmpElement = [0] * len(labelTypeList[n])
			tmpElement[labelTypeList[n].index(element)] = 1
			for e in tmpElement:
				res.append(e)
		else:
			res.append(element)
	return res

def dot(a, b): #calculate dot product of two vectors
	if len(a)!=len(b):
		print('dimension of 2 vectors are not matching!')
		exit()
	ret = 0
	for i in range(len(a)):
		ret += a[i] * b[i]
	return ret

def matrixMul(vec, matrix):
	res = []
	tempRes = 0
	for m in range(len(matrix)):
		res.append(dot(matrix[m], vec))
	return res

def calcCrossEntropy(output, instance):
	y = instance[-1]
	E =  -y * log(output) - (1- y) * log(1 - output)
	return E

def initWeights(numhUnits, featureLen):
	hidenWeights = []
	outputWeights = []
	if numhUnits > 0:
		for m in range(numhUnits):
			tmpColumn = []
			for n in range(featureLen):
				tmpColumn.append(random.uniform(-0.01, 0.01))
			hidenWeights.append(tmpColumn)
	
		for m in range(numhUnits + 1):
			outputWeights.append(random.uniform(-0.01, 0.01))
	else:
		for m in range(featureLen):
			outputWeights.append(random.uniform(-0.01, 0.01))
	return hidenWeights, outputWeights

def deltaOuput(output, instance):
	deltaW = 0
	y = instance[-1]
	deltaW = output - y
	return deltaW

def deltaHiden(outputHiden, deltaW, outputWeights):
	deltaH = []
	if len(outputHiden) != len(outputWeights):
		print('Dimension not matching!')
		exit()
	for i in range(len(outputWeights)):
		weights = outputWeights[i]
		hiden_output = outputHiden[i]
		deltaH.append(hiden_output * (1 - hiden_output) * deltaW * weights)
	return deltaH

def updateWHO(output, deltaW, outputWeights, learningRate): 
#give output for this layer, and delta calc from the output layer
#and we assume the output and outputWeight has the same dimension
	for h in range(len(outputWeights)):
		outputWeights[h] -= output[h] * deltaW * learningRate
	return outputWeights

def updateWIH(instance, deltaH, hidenWeights, learningRate):
	deltaH = deltaH[1:] #we remove the error at bias since it makes no sense for updata input layer
	for i in range(len(deltaH)):
		for j in range(len(hidenWeights[i])):
			hidenWeights[i][j] -= deltaH[i] * learningRate * instance[j]
	return hidenWeights

def standardize(data_set, numericalIdx):
	for idx in numericalIdx:
		mu = 0
		sigma = 0
		for data in data_set:
			mu += data[idx]
		mu = mu / len(data_set)
		for data in data_set:
			sigma += math.pow(data[idx] - mu, 2)
		sigma = sqrt(sigma / len(data_set))
		for data in data_set:
			data[idx] = (data[idx] - mu) / sigma
	return data_set

#test code
labels, dataSet, labelTypeList = preProcessingData(trainFile)
labels, testSet, labelTypeList = preProcessingData(testFile)

#standardize each numerical feature
numericalIdx = []
for i in range(len(dataSet[0])):
	if isinstance(dataSet[0][i], float):
		numericalIdx.append(i)

#begin standardization
dataSet = standardize(dataSet, numericalIdx)
testSet = standardize(testSet, numericalIdx)
trainSet = copy.deepcopy(dataSet)

classType = labelTypeList[-1]
tmp = getOneOfk(dataSet[0][0:-1], labelTypeList)

num_pos_train = 0
num_neg_train = 0
num_pos_test = 0
num_neg_test = 0

for trInstance in dataSet:
	if trInstance[-1] == classType[0]: #negative or equal
		num_neg_train += 1
	elif trInstance[-1] == classType[1]:
		num_pos_train += 1

for teInstance in testSet:
	if teInstance[-1] == classType[0]: #negative or equal
		num_neg_test += 1
	elif teInstance[-1] == classType[1]:
		num_pos_test += 1

Wh, Wo = initWeights(h, len(tmp) + 1)
random.shuffle(trainSet)

for step in range(e):
	#Initialize the problem
	for dataInstance in trainSet:
		tmpDataInstance = dataInstance[0:-1]
		trainInstance = getOneOfk(tmpDataInstance, labelTypeList)
		trainInstance.insert(0, 1)
		if dataInstance[-1] == classType[0]:
			trainInstance.append(0)
		else:
			trainInstance.append(1)

		if h != 0: #with hiden layer
			#forward propagation
			hidenLayer_input = matrixMul(trainInstance[0:-1], Wh)
			hidenLayer_output = sigmoid(hidenLayer_input)
			hidenLayer_output.insert(0, 1)
			outLayer_input = dot(hidenLayer_output, Wo)
			outLayer_output = sigmoid(outLayer_input)

			#back propagation
			outputErr = deltaOuput(outLayer_output, trainInstance)
			hidenErr = deltaHiden(hidenLayer_output, outputErr, Wo)
			Wo = updateWHO(hidenLayer_output, outputErr, Wo, l)
			Wh = updateWIH(trainInstance[0:-1], hidenErr, Wh, l)
		else: #without hiden layer, input connected directly to the output layer
			outLayer_input = dot(trainInstance[0:-1], Wo)
			outLayer_output = sigmoid(outLayer_input)
			outputErr = deltaOuput(outLayer_output, trainInstance)
			Wo = updateWHO(trainInstance[0:-1], outputErr, Wo, l)

roc_instance_train = []
roc_instance_test = []

TP = 0
FP = 0
last_TP = 0
FPR_train = []
TPR_train = []
last_conf = 0
for ins in dataSet:
	tmpIns = ins[0:-1]
	trainIns = getOneOfk(tmpIns, labelTypeList)
	trainIns.insert(0, 1)
	if ins[-1] == classType[0]:
		trainIns.append(0)
	else:
		trainIns.append(1)

	if h != 0:
		#forward propagation
		for_calc_h = matrixMul(trainIns[0:-1], Wh)
		for_calc_h = sigmoid(for_calc_h)
		for_calc_h.insert(0, 1)
		for_calc_o = dot(for_calc_h, Wo)
		for_calc_o = sigmoid(for_calc_o)
	else:
		for_calc_o = dot(trainIns[0:-1], Wo)
		for_calc_o = sigmoid(for_calc_o)
	roc_instance_train.append([ins[-1], for_calc_o])
roc_instance_train.sort(key=operator.itemgetter(1), reverse=True)

for pairs in roc_instance_train:
	confidence = pairs[1]
	classLabel = pairs[0]
	#for ROC curve
	if last_conf != confidence and classLabel == classType[0] and TP > last_TP:
		FPR = FP / num_neg_train
		TPR = TP / num_pos_train
		FPR_train.append(FPR)
		TPR_train.append(TPR)
		last_TP = TP
	if classLabel == classType[1]:
		TP += 1
	else:
		FP += 1
	last_conf = confidence
FPR = FP / num_neg_train
TPR = TP / num_pos_train
FPR_train.append(FPR)
TPR_train.append(TPR)

TP = 0
FP = 0
last_TP = 0
FPR_test = []
TPR_test = []
last_conf = 0
for n in range(len(testSet)):
	testInstance = testSet[n]
	tmpIns_test = testInstance[0:-1]
	testIns = getOneOfk(tmpIns_test, labelTypeList)
	testIns.insert(0, 1)
	if testInstance[-1] == classType[0]:
		testIns.append(0)
	else:
		testIns.append(1)
	if h != 0:
	#forward propagation
		for_test_h = matrixMul(testIns[0:-1], Wh)
		for_test_h = sigmoid(for_test_h)
		for_test_h.insert(0, 1)
		for_test_o = dot(for_test_h, Wo)
		for_test_o = sigmoid(for_test_o)
	else:
		for_test_o = dot(testIns[0:-1], Wo)
		for_test_o = sigmoid(for_test_o)
		#for ROC curve
	roc_instance_test.append([testInstance[-1], for_test_o])
roc_instance_test.sort(key=operator.itemgetter(1), reverse=True)

for pair in roc_instance_test:
	confidence = pair[1]
	classLabel = pair[0]
	if last_conf != confidence and classLabel == classType[0] and TP > last_TP:
		FPR = FP / num_neg_test
		TPR = TP / num_pos_test
		FPR_test.append(FPR)
		TPR_test.append(TPR)
		last_TP = TP
	if classLabel == classType[1]:
		TP += 1
	else:
		FP += 1
	last_conf = confidence
FPR = FP / num_neg_test
TPR = TP / num_pos_test
FPR_test.append(FPR)
TPR_test.append(TPR)

print FPR_train
p
plt.figure(1)
plt.plot(FPR_train, TPR_train, color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for training set')

plt.figure(2)
plt.plot(FPR_test, TPR_test, color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for test set')

plt.show()
