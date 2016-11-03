import re
import sys
from heapq import *
import math

feature_name = []
actual = [] #test set label
# class_name = []
isClassify = 0
featureNum = 0;
train_set = []
test_set = []
response = [] # train set label
instance2index = {}
K = 3
name2index = {}
TEST = 0

def isfloat(x):
	try:
		float(x)
		return True
	except ValueError:
		return False

def load_arff(data_name, isTrain):
	global feature_name, isClassify, response, actual, name2index
	data = []
	enterData = 0
	with open(data_name) as file:
		count = 0
		for line in file:
			vec = re.split("[ ,}{\n\r']+", line)
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
							class_name = vec[2:]
							for k in range(len(class_name)):
								name2index[class_name[k]] = k

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
		cur_dis = math.sqrt(cur_dis)
		tup = (-cur_dis, response[i])
		if len(h) < K:
			heappush(h, tup)
		else:
			tmp_top = heappop(h)
			if tmp_top[0] >= tup[0]: #negative number
				heappush(h, tmp_top)
			else:
				heappush(h, tup)
			# heappushpop(h, tup)
		#store nearest K points

	res = {} # map from class to number, class : (appearance, distance)
	count = K
	if isClassify:
		while count != 0:
			tup = heappop(h)

			if tup[1] in res:
				res[tup[1]][0] += 1 #appearance
				res[tup[1]][1] = -tup[0] #distance
			else:
				res[tup[1]] = [1, -tup[0]]
			count -= 1

		maxApperance = 0
		minDis = 100000000
		cur_prediction = 0
		for ele in res.items():
			# if ele[1][0] > maxApperance or (ele[1][0] == maxApperance and ele[1][1] < minDis) or (ele[1][0] == maxApperance and ele[1][1] == minDis and name2index[ele[0]] < name2index[cur_prediction]):
			if ele[1][0] > maxApperance \
			or (ele[1][0] == maxApperance and name2index[ele[0]] < name2index[cur_prediction]):
				maxApperance = ele[1][0]
				minDis = ele[1][1]
				cur_prediction = ele[0]
		return cur_prediction
	else:
		mean = 0
		while count != 0:
			tup = heappop(h)
			# print type(tup[1]) 
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
	else:
		argv = sys.argv;
		argvNum = len(argv)
		train_name = argv[1]
		test_name = argv[2]
		K = int(argv[3])

	train_set = load_arff(train_name, 1)
	test_set = load_arff(test_name, 0)
	featureNum = len(feature_name)

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
		format_precision = float(correct) / len(prediction)
		print ("Accuracy : " + str("{0:.16f}".format(format_precision))),
	else:
		print "Mean absolute error : " + str("{0:.16f}".format(error))
		print ("Total number of instances : " + str(len(prediction))),

if __name__ == "__main__":
	main()