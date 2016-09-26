import re
import math
import csv
import Queue
import sys
import copy
import random
import matplotlib.pyplot as plt


train_data = [];
test_data = [];
label =[];
thresholdNum = []
attributeNum = 0;
instanceNum = 0;
nominal = {}
stopThreshold = 4;
thresholdMap = []
MAX_INIT = -2147483648
MIN_INIT = 2147483647
median = 1
STOPMEG = 0
def getPrediction(node):
	p = 0
	n = 0
	if node == 0:
		return 0
	currentSet = node.currentSet
	for i in currentSet:
		if label[i] == 0:
			n += 1
		else:
			p += 1
	if n > p:
		return 0
	elif n < p:
		return 1
	else:
		return getPrediction(node.parent)


class node:
	def __init__(self, visited, currentSet, parent):
		# print (" ")
		# print ("size " + str(len(currentSet)))
		self.isLeaf = 0
		self.visited = visited
		self.index = -1
		self.threshold = -1
		self.gainVal = 0
		self.prediction = 0
		self.parent = parent
		self.currentSet = currentSet
		self.child = []
		

		if self.judgeStop(visited[:], currentSet):
			self.isLeaf = 1
			self.prediction = getPrediction(self)
			return
		gain = self.inforgain(visited[:], currentSet)
		#if no information gain, then return
		if gain <= 0:
			if STOPMEG:
				print "no more information gain in any node"
			self.isLeaf = 1
			self.prediction = getPrediction(self)
			return 
		
		if nominal[self.index] != "NUMERIC":
					# nominal
			self.isNominal = 1
		else:
			self.isNominal = 0


		self.visited.append(self.index);

		if not self.isNominal:
			[left, right] = self.getChild(currentSet, self.index, self.threshold)
			self.child.append(node(self.visited[:], left, self))
			self.child.append(node(self.visited[:], right, self))
		else:
			tmp_child = self.getChild(currentSet, self.index, self.threshold)
			for eachChild in tmp_child:
				self.child.append(node(self.visited[:], eachChild, self))
		return

	def predict(self, instance):
		if self.isLeaf:
			return self.prediction
		else:
			branch = self.predict_helper(instance)
			return self.child[branch].predict(instance)
	def predict_helper(self, instance):
		if self.isNominal:
			return thresholdMap[self.index][instance[self.index]]
		else:
			if instance[self.index] > self.threshold:
				return 1
			else:
				return 0
	def getChild(self, currentSet, index, threshold):
		if not self.isNominal:
			left = []
			right = []
			for i in currentSet:
				if train_data[i][index] > threshold:
					right.append(i)
				else:
					left.append(i)
			return [left, right]
		else:
			#let thresholdMap be a dict, nominal -> number
			childList = [[] for i in range(thresholdNum[self.index])]
			for i in currentSet:
				feature_belong_class = thresholdMap[index][train_data[i][index]]
				childList[feature_belong_class].append(i)
			return childList
			
	def inforgain(self, visited, currentSet):
			max_val = MAX_INIT
			index = -1
			for attribute_index in range(attributeNum):
				if visited.count(attribute_index) != 0:
					continue
				if nominal[attribute_index] == "NUMERIC":
					vec = self.helper_numeric(attribute_index, currentSet[:])
					val = vec[0]
					mid = vec[1]
					self.threshold = mid
				else:
					val = self.helper_nominal(attribute_index, currentSet[:])
					self.threshold = None
				#val is negative
				# print ("val is " + str(val))
				if val > max_val:
					max_val = val
					index = attribute_index
			#calculate self entropy to get the information gain by minus the entropy of child
			p = 0
			n = 0
			for i in currentSet:
				if label[i] == 1:
					p += 1
				else:
					n += 1
			pn = p + n + 0.0
			if p == pn or n == pn:
				parent_entropy = 0
			else:
				parent_entropy = -(p / pn * math.log(p / pn)) - (n / pn * math.log(n / pn))

			gain = parent_entropy + max_val

			if gain <= 0:
				return gain

			visited.append(index)
			self.index = index
			return gain
	


	def judgeStop(self, visited, currentSet):
		#no more split candidate
		if len(visited) == attributeNum:
			if STOPMEG:
				print ("no more split candidate")
			return 1

		#have less than m instances
		if len(currentSet) < stopThreshold:
			if STOPMEG:
				print "have less than m instances"
			return 1

		#has only one class
		l = []
		for i in currentSet:
			if l.count(label[i]) != 0:
				l.append(label[i])
		if len(l) == 1:
			if STOPMEG:
				print "has only one class remaining"
			return 1

		return 0

	def numeric_info(self, vec, mid, currentSet):
		length = len(vec)
		falseNeg = 0
		truePos = 0
		falsePos = 0
		trueNeg = 0
		for i in range(length):
			if vec[i] <= mid:
				if label[currentSet[i]] == 0:
					trueNeg += 1
				else:
					falseNeg += 1
			else:
				if label[currentSet[i]] == 0:
					falsePos += 1
				else:
					truePos += 1
		#plogp calculate the max
		pNeg = (trueNeg + falseNeg + 0.0) / length
		pPos = 1 - pNeg

		if pNeg == 1 or pPos == 1:
			return 0


		acc = 0.0
		t1 = trueNeg / (trueNeg + falseNeg + 0.0)
		t2 = 1 - t1
		if t1 != 1 and t1 != 0:
			acc += pNeg * ( t1 * math.log(t1) + t2 * math.log(t2) )


		t1 = truePos / (truePos + falsePos + 0.0)
		t2 = 1 - t1
		if t1 != 1 and t1 != 0:
			acc += pPos * ( t1 * math.log(t1) + t2 * math.log(t2) )
		# -1 ~ 0
		return acc
	def helper_numeric(self, index, currentSet):
		#find the best split in a particular feature
		# print ("numeric :: current set size is : " + str(len(currentSet)))

		vec = []
		for instance_index in currentSet:
			vec.append(train_data[instance_index][index])
		l = len(currentSet)
		min_val = MIN_INIT
		max_val = MAX_INIT
		max_entropy = -1


		if median:
			currentSet.sort()
			tmp = currentSet[len(currentSet) / 2]
			mid = train_data[tmp][index]
			# print mid
		else:
			for i in currentSet:
				cur = train_data[i][index]
				if cur > max_val:
					max_val = cur
				if cur < min_val:
					min_val = cur
			mid = (max_val + min_val) / 2
		max_entropy = self.numeric_info(vec, mid, currentSet)


		# for i in range(l - 1):
		# 		instance_index = currentSet[i];
		# 		second = currentSet[i + 1]
		# 		# print str(instance_index) + "   " + str(second)
		# 		mid = (train_data[instance_index][index] + train_data[second][index]) / 2
		# 		max_entropy = max(max_entropy, self.numeric_info(vec, mid, currentSet) )

		return [max_entropy, mid]

	def getEachEntropy(self, index, row):
		p = 0
		n = 0
		for ele in row:
			if label[ele] == 0:
				n += 1
			else:
				p += 1
		l = len(row)
		if p == l or n == l:
			return 0
		else:
			l += 0.0
			t1 = p / l
			t2 = n / l
			return t1 * math.log(t1) + t2 * math.log(t2)

	def nominal_info(self, vec, index, currentSet):
		acc = [[] for i in range( len(nominal[index]) )]
		count = 0
		l = len(vec)
		for count in range(l):
			# print "map is    "
			# print "          " + str(thresholdMap[index])
			cur = thresholdMap[index][vec[count]]
			acc[cur].append(currentSet[count])
		
		entropy = 0
		l += 0.0
		for row in acc:
			cur_l = len(row)
			if cur_l == 0:
				continue
			entropy += cur_l / l * self.getEachEntropy(index, row)
		return entropy

	def helper_nominal(self, index, currentSet):
		vec = []
		# print ("current set size is : " + str(len(currentSet)))
		for instance_index in currentSet:
			vec.append(train_data[instance_index][index])
		l = len(currentSet)
		entropy = self.nominal_info(vec, index, currentSet)
		return entropy
	
def check(q):
	if q.empty():
		return
	qq = Queue.Queue()

	s = ""
	while not q.empty():
		tmp = q.get()
		l = len(tmp.child)
		if tmp.index == -1:
			s += "NULL"
		else:
			s += str(tmp.index)
		s += "    "
		for i in range(l):
			qq.put(tmp.child[i])
	# print "      " + s
	check(qq)
def func(local_train_data, local_test_data, trainSetSize):
	global train_data, test_data, label, stopThreshold , attributeNum, instanceNum, nominal, label, thresholdMap, thresholdNum
	accuracy = []
	copy_train_data = copy.deepcopy(local_train_data)
	copy_test_data = copy.deepcopy(local_test_data)
	for train_time in range(10):
		train_data = copy.deepcopy(copy_train_data)
		test_data = copy.deepcopy(copy_test_data)
		label = []
		attributeNum = len(train_data["attributes"]) - 1
		if attributeNum < 0:
			return

		#thresholdmap store the map from value to int for each feature
		thresholdMap = [{} for i in range(attributeNum)]

		#nominal stores the values of each features
		count = 0
		cur_row = train_data["data"][0]
		for i in range(attributeNum):
			ele = cur_row[i]
			if not type(ele) == unicode:
				nominal[count] = "NUMERIC"
			else:
				nominal[count] = []
				#train_data["attribuets"] is a tuple, two element, first name ,second list containing values
				for instance in train_data["attributes"][count][1]:
					if nominal[count].count(instance) == 0:
						nominal[count].append(instance)
			count += 1

		#build thresholdmap
		for ele in nominal:
			count = 0
			if nominal[ele] != "NUMERIC":
				for item in nominal[ele]:
					thresholdMap[ele][item] = count
					count += 1

		map_label = {}
		count = 0

		#map label to 0 and 1
		
		thresholdNum = [0 for i in range(attributeNum)]
		for i in range(attributeNum):
			if not type(train_data["data"][0][i]) is unicode:
				continue
			thresholdNum[i] = len(nominal[i])
				
		visited = []
		

		candidate_train_data = train_data["data"]
		test_data = test_data["data"]

		train_length = len(candidate_train_data)
		selected = []
		selectNum = 0
		while selectNum < train_length * trainSetSize:
			cur = random.randint(0, train_length - 1)
			if selected.count(cur) == 0:
				selected.append(cur)
				selectNum += 1
		tmp_train_data = []
		for ele in selected:
			tmp_train_data.append(candidate_train_data[ele])
		train_data = tmp_train_data

		for ele in train_data:
			if ele[attributeNum] not in map_label:
				map_label[ele[attributeNum]] = count
				count += 1
			label.append(map_label[ele[attributeNum]])


		instanceNum = len(train_data)
		currentSet = [i for i in range(instanceNum)]

		root = node(visited, currentSet, 0)

		# print ("-------------------------------------------------------------------------------------------")

		q = Queue.Queue()
		q.put(root)
		check(q)


		test_label = []
		predict_label = []
		for ele in test_data:
			test_label.append(map_label[ele[attributeNum]])
			predict_label.append(root.predict(ele))

		test_num = len(test_label)
		correct = 0
		for i in range(test_num):
			if test_label[i] == predict_label[i]:
				correct += 1
			# print str(predict_label[i]) +"      "  + str(test_label[i])
		# print str(correct) +"      "  + str(test_num)
		correct += 0.0
		accuracy.append(correct / test_num)
		if trainSetSize == 1:
			break

	sumAccuracy = 0
	for ele in accuracy:
		sumAccuracy += ele
	avgAccuracy = sumAccuracy / len(accuracy)
	# print "accuracy:"
	# print sumAccuracy
	return avgAccuracy
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
def load_arff(data_name):
	data = {}
	data["attributes"] = []
	data["data"] = []
	enterData = 0
	with open(data_name) as file:
		count = 0
		for line in file:
			if enterData:
				vec = re.split("[,\n ]+", line)
				l = len(vec)
				for i in range(l):
					if isfloat(vec[i]):
						vec[i] = float(vec[i])
				data["data"].append(vec)
				continue
			vec = re.split("[ ,}\n']+", line)
			if len(vec) > 0 and vec[0] == "@attribute":
				if vec[2] == "{":
					values = []
					for i in range(2, len(vec)):
						if i == len(vec) - 1:
							vec[i] = vec[i][:-1]
						if vec[i] != "{" and vec[i] != "}" and vec[i] != "":
							values.append(vec[i])
					data["attributes"].append((vec[1], values))

				else:
					data["attributes"].append((vec[1], "NUMERIC"))

			if vec[0] == "@data":
				enterData = 1

	return data
def main():
	global stopThreshold
	#dt-learn.py 
	# argv = sys.argv
	# argvNum = len(argv)
	# if argvNum != 4:
	# 	print "error arguments number"
	# 	return
	# train_data_name = argv[1]
	# test_data_name = argv[2]
	# stopThreshold = int(argv[3])
	# train_data = arff.load(open(train_data_name))
	# test_data = arff.load(open(test_data_name))

	sizeVec = [2, 5, 10, 20]

	#train_data = load_arff('data/diabetes_train.arff')
	# test_data = load_arff('data/diabetes_test.arff')

	train_data = load_arff('data/heart_train.arff')
	test_data = load_arff('data/heart_test.arff')
	avgAcc = []
	minAcc = []
	maxAcc = []

	for ele in sizeVec:
		stopThreshold = ele
		tmp = func(train_data, test_data, 1)
		avgAcc.append(tmp)
	plt.subplot(111)
	plt.plot(sizeVec, avgAcc, label = "average")

	plt.axis([1, 21, 0.5, 1])
	plt.ylabel("accuracy")
	plt.xlabel("stopping criteria(m)")
	plt.show()


if __name__ == "__main__":
	main()