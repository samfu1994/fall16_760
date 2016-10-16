import re
import sys
from heapq import *

feature_name = []
actual = [] #test set label
# class_name = []
isClassify = 0
featureNum = 0;
train_set = []
test_set = []
response = [] # train set label
K = 1
kk = [0 for i in range(3)]
TEST = 0
def isfloat(x):
	try:
		float(x)
		return True
	except ValueError:
		return False

def load_arff(data_name, isTrain):
	global feature_name, isClassify, response, actual
	data = []
	enterData = 0
	with open(data_name) as file:
		count = 0
		for line in file:
			vec = re.split("[ ,}{\n']+", line)
			if enterData:
				#reach data section
				row = []
				vec = filter(None, vec)
				if isTrain:
					if isfloat(vec[-1]):
						response.append(float(vec[-1]))
					else:
						response.append(vec[-1])
				else:
					if isfloat(vec[-1]):
						actual.append(float(vec[-1]))
					else:
						actual.append(vec[-1])

				vec = vec[:-1]
				l = len(vec)
				for i in range(l):
					row.append(float(vec[i]))
				data.append(row)
			
			else :
				# not reach data section
				if isTrain:
					vec = filter(None, vec)
					if len(vec) > 0 and vec[0] == "@attribute":
						if vec[1] == "class" :
							isClassify = 1
							# class_name = vec[2:]
						elif vec[1] != "response":
							feature_name.append(vec[1])

			if vec[0] == "@data":
				enterData = 1
	return data

def predict(cur):
	l = len(train_set) 
	h = []
	for i in range(l):
		cur_dis = 0
		for j in range(featureNum):
			tmp = cur[j] - train_set[i][j]
			cur_dis += tmp * tmp
		tup = (-cur_dis, response[i])
		if len(h) < K:
			heappush(h, tup)
		else:
			heappushpop(h, tup)
	res = {} # map from class to number
	count = K
	if isClassify:
		while count != 0:
			tup = heappop(h)

			if tup[1] in res:
				res[tup[1]] += 1
			else:
				res[tup[1]] = 1
			count -= 1
		maxApperance = 0
		
		count = 0
		cur_prediction = 0
		for ele in res.items():
			count+= 1
			if ele[1] > maxApperance:
				maxApperance = ele[1]
				cur_prediction = ele[0]
		return cur_prediction
	else:
		mean = 0
		while count != 0:
			tup = heappop(h)
			mean += tup[1]

			count -= 1
		mean /= K

		return mean
def main():
	# load_arff("wine_train.arff")
	global K, train_set, test_set, featureNum


	if TEST:
		train_name = "yeast_train.arff"
		test_name = "yeast_test.arff"
		# train_name = "wine_train.arff"
		# test_name = "wine_test.arff"
		kk[0] = 1
		kk[1] = 5
		kk[2] = 10   
	else:
		argv = sys.argv;
		argvNum = len(argv)
		train_name = argv[1]
		test_name = argv[2]
		kk[0] = int(argv[3])
		kk[1] = int(argv[4])
		kk[2] = int(argv[5])

	accVec = []

	origin_train_set = load_arff(train_name, 1)
	origin_test_set = load_arff(test_name, 0)

	cross_acc = [0 for i in range(3)]
	cross_error = [0 for i in range(3)]

	for k in range(3): # for k1 k2 k3
		correct = 0
		error = 0;
		prediction = []
		K = kk[k]
		for i in range(len(origin_train_set)): # cross validation
			cur_index = i
			if cur_index > 0:
				train_set = origin_train_set[0 : cur_index]
				train_set += origin_train_set[cur_index + 1 : ]
			else:
				train_set = origin_train_set[cur_index + 1 : ]
			test_set = origin_train_set[cur_index]

			featureNum = len(feature_name)

			prediction.append(predict(test_set))
			if isClassify:
				if response[i] == prediction[i]:
					correct += 1
			else:
				error += abs(prediction[i] - response[i])

		cross_acc[k] = float(correct) / len(origin_train_set)
		cross_error[k] = float(error) / len(origin_train_set)


	maxval = 0
	minerr = 100000000
	for i in range(3):
		if isClassify:
			if cross_acc[i] > maxval:
				maxval = cross_acc[i]
				K = kk[i]
		else:
			if cross_error[i] < minerr:
				minerr = cross_error[i]
				K = kk[i]

	train_set = origin_train_set
	test_set = origin_test_set

	prediction = []

	for i in range(len(test_set)):
		prediction.append(predict(test_set[i]))


	print "k value : " + str(K)
	correct = 0
	error = 0
	for i in range(len(prediction)):
		if isClassify:
			if actual[i] == prediction[i]:
				correct += 1
		else:
			error += abs(prediction[i] - actual[i])
		if not isClassify:
			print "Predicted value : " + str("{0:.6f}".format(prediction[i])) + "	Actual value : " + str("{0:.6f}".format(actual[i]))
		else:
			print "Predicted class : " + str(prediction[i]) + "	Actual class : " + str(actual[i])

	error /= len(prediction)
	if isClassify:
		print "Number of correctly classified instances : " + str(correct)
		print "Total number of instances : " + str(len(prediction))
		print "Accuracy : " + str(float(correct) / len(prediction))
	else:
		print "Mean absolute error : " + str("{0:.16f}".format(error))
		print "Total number of instances : " + str(len(prediction))


if __name__ == "__main__":
	main()