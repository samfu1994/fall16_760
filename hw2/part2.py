import re
import sys
from heapq import *
import matplotlib.pyplot as plt

feature_name = []
actual = [] #test set label
# class_name = []
isClassify = 0
featureNum = 0;
train_set = []
test_set = []
response = [] # train set label
K = 1
TEST = 1
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
	# print cur
	for i in range(l):
		cur_dis = 0
		# print train_set[i]
		for j in range(featureNum):
			tmp = cur[j] - train_set[i][j]
			cur_dis += tmp * tmp
			# print cur[j], train_set[i][j]
		# print "this : " + str(cur_dis) + "   " + response[i]
		tup = (-cur_dis, response[i])
		if len(h) < K:
			heappush(h, tup)
		else:
			heappushpop(h, tup)
		# print h
	# print ""
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
			# print type(tup[1]) 
			mean += tup[1]

			count -= 1
		mean /= K

		return mean
def main():
	Kvec = [1, 5, 10, 20, 30]
	global K, train_set, test_set, featureNum


	if TEST:
		train_name = "yeast_train.arff"
		test_name = "yeast_test.arff"
		# train_name = "wine_train.arff"
		# test_name = "wine_test.arff"
	else:
		argv = sys.argc;
		argvNum = len(argv)
		train_name = argv[1]
		test_name = argv[2]
		K = int(argv[3])
	train_set = load_arff(train_name, 1)
	test_set = load_arff(test_name, 0)

	featureNum = len(feature_name)
	accVec  = []
	errVec = []
	for i in range(len(Kvec)):
		K = Kvec[i]
		prediction = []
		for i in range(len(test_set)):
			prediction.append(predict(test_set[i]))
		# print type(train_set[0][0])
		print K
		correct = 0
		error = 0
		for i in range(len(prediction)):
			if isClassify:
				if actual[i] == prediction[i]:
					correct += 1
			else:
				error += abs(prediction[i] - actual[i])
		errVec.append(float(error) / len(prediction))
		accVec.append(float(correct) / len(prediction) )
	print accVec
	print errVec

	plt.subplot(111)
	plt.plot(Kvec, accVec, label = "accuracy")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
	plt.axis([0, 31, 0, 1])
	plt.ylabel("accuracy")
	plt.xlabel("K")
	plt.show()

if __name__ == "__main__":
	main()