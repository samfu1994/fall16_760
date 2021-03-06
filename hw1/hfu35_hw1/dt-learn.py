import math
import csv
import Queue
import sys
import copy
import re
import random
from sets import Set
import collections

train_data = [];
test_data = [];
label =[];
thresholdNum = []
attributeNum = 0;
instanceNum = 0;
nominal = {}
stopThreshold = 2;
trainSetSize = 1;
originAttributes = 0
thresholdMap = []
MAX_INIT = -2147483648
MIN_INIT = 2147483647
median = 0
RANDINT = 20000
STOPMEG = 0
inverseThresholdMap = {}
inverseMap = []
map_label = {}


splitter = "|	"


def getPrediction(node):
	#get prediction according to the p or n which one has more instances currently
	p = 0
	n = 0
	if isinstance(node, int):
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
		#build tree using init function
		self.isLeaf = 0
		self.visited = visited
		self.index = -1
		self.threshold = -1
		self.gainVal = 0
		self.prediction = 0
		self.parent = parent
		self.currentSet = currentSet
		self.child = []
		self.p = 0
		self.n = 0
		for ele in currentSet:
			if label[ele] == 0:
				self.n += 1
			else:
				self.p += 1

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
			self.isNominal = 1
		else:
			self.isNominal = 0

		if not self.isNominal:
			[left, right] = self.getChild(currentSet, self.index, self.threshold)
			self.child.append(node(self.visited[:], left, self))
			self.child.append(node(self.visited[:], right, self))
		else:
			tmp_child = self.getChild(currentSet, self.index, self.threshold)
			count = 0
			for eachChild in tmp_child:
				self.child.append(node(self.visited[:], eachChild, self))
				count += 1
		return

	def predict(self, instance):
		#api for testing set, return when reach leaf
		if self.isLeaf:
			return self.prediction
		else:
			branch = self.predict_helper(instance)
			return self.child[branch].predict(instance)

	def predict_helper(self, instance):
		#call by predict when encouter an internal node, keep going downward
		if self.isNominal:
			return thresholdMap[self.index][instance[self.index]]
		else:
			if instance[self.index] > self.threshold:
				return 1
			else:
				return 0


	def getChild(self, currentSet, index, threshold):
		#create child nodes according to threshold
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
			self.threshold = [0 for i in range(thresholdNum[self.index])]
			childList = [[] for i in range(thresholdNum[self.index])]
			for i in currentSet:
				feature_belong_class = thresholdMap[index][train_data[i][index]]
				childList[feature_belong_class].append(i)
				self.threshold[feature_belong_class] = train_data[i][index]
			for i in range(len(self.threshold)):
				if self.threshold[i] == 0:
					self.threshold[i] = nominal[index][i]
			return childList
			
	def inforgain(self, visited, currentSet):
		#return the max information gain of this node, set the index and threshold of this node
			max_val = MAX_INIT
			max_mid = 0
			index = -1
			for attribute_index in range(attributeNum):
				if visited.count(attribute_index) != 0:
					continue
				if nominal[attribute_index] == "NUMERIC":
					vec = self.helper_numeric(attribute_index, currentSet[:])
					val = vec[0]
					mid = vec[1]
				else:
					val = self.helper_nominal(attribute_index, currentSet[:])
					mid = 0
				#val is negative, plogp is smaller than 0 because p is smaller than 1
				if val > max_val + 0.000000000000001:
					max_val = val
					index = attribute_index
					max_mid = mid
			#get the split with the least entropy
			if nominal[index] == "NUMERIC":
				self.threshold = max_mid
			else:
				self.threshold = []
				for i in range(thresholdNum[self.index]):
					self.threshold.append(inverseThresholdMap[self.index][i]) 
			
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
			if l.count(label[i]) == 0:
				l.append(label[i])
		if len(l) <= 1:
			if STOPMEG:
				print "has only one class remaining"
			return 1

		return 0

	def numeric_info(self, vec, mid, currentSet):
		#call by helper_numeric to calculate
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

		acc = 0.0

		if trueNeg + falseNeg == 0:
			t1 = 0
		else:
			t1 = trueNeg / (trueNeg + falseNeg + 0.0)
		t2 = 1 - t1

		if t1 != 1 and t1 != 0:
			acc += pNeg * ( t1 * math.log(t1))
		if t2 != 1 and t2 != 0:
			acc +=  pNeg * ( t2 * math.log(t2))


		if truePos + falsePos == 0:
			t1 = 0
		else:
			t1 = truePos / (truePos + falsePos + 0.0)
		t2 = 1 - t1

		if t1 != 1 and t1 != 0:
			acc += pPos * ( t1 * math.log(t1))
		if t2 != 1 and t2 != 0:
			acc += pPos * t2 * math.log(t2)
		# -1 ~ 0
		return acc

	# def getMidVec(self, valueSet):
	# 	#get the split point of a numeric feature
	# 	l = len(valueSet)
	# 	s = Set()
	# 	res = []
	# 	pre_neg = -1
	# 	pre_pos = -1
	# 	for ele in valueSet:
	# 		if ele not in s:
	# 			s.add(ele)
	# 	s = sorted(s, key = lambda ele : ele[0])
	# 	for i in range(len(s)):
	# 		cur = s[i][0]
	# 		cur_label = s[i][1]
	# 		for j in range(i + 1, len(s)):
	# 			if cur == s[j][0]:
	# 				continue
	# 			if cur_label == s[j][1]:
	# 				if j < len(s) - 1 and s[j + 1][0] == s[j][0] and s[j + 1][1] != s[j][1]:
	# 					res.append((cur + s[j][0]) / 2)
	# 				break
	# 			res.append((cur + s[j][0]) / 2)

	# 	return res

	# def helper_numeric(self, index, currentSet):
	# 	#find the best split in a particular feature
	# 	vec = []
	# 	for instance_index in currentSet:
	# 		vec.append(train_data[instance_index][index])
	# 	l = len(currentSet)
	# 	min_val = MIN_INIT
	# 	max_val = MAX_INIT
	# 	max_entropy = -1
	# 	max_mid = -1

	# 	valueSet = [];
	# 	for ele in currentSet:
	# 		valueSet.append((train_data[ele][index], label[ele]))
	# 	valueSet.sort(key = lambda tup: tup[0]);

	# 	count = [0, 0]
	# 	for i in range(len(currentSet)):
	# 		count[label[currentSet[i]]] += 1
	# 	if median:
	# 		currentSet.sort()
	# 		tmp = currentSet[len(currentSet) / 2]
	# 		max_mid = train_data[tmp][index]
	# 	else:
	# 		midVec = self.getMidVec(valueSet)
	# 		ll = len(midVec)
	# 		for i in range(ll):

	# 			mid = midVec[i]
	# 			t = self.numeric_info(vec, mid, currentSet)
	# 			if max_entropy < t:
	# 				max_entropy = t
	# 				max_mid = mid

	# 	return [max_entropy, max_mid]


	def getMidVec(self, valueSet):
		#get the split point of a numeric feature
		res = []
		l = len(valueSet)
		for i in range(l - 1):
			res.append((valueSet[i] + valueSet[i + 1]) / 2)

		return res

	def helper_numeric(self, index, currentSet):
		#find the best split in a particular feature
		vec = []
		for instance_index in currentSet:
			vec.append(train_data[instance_index][index])
		l = len(currentSet)
		min_val = MIN_INIT
		max_val = MAX_INIT
		max_entropy = -1
		max_mid = -1

		valueSet = [];
		for ele in currentSet:
			if valueSet.count(train_data[ele][index]) == 0:
				valueSet.append(train_data[ele][index])
		valueSet.sort();

		count = [0, 0]
		for i in range(len(currentSet)):
			count[label[currentSet[i]]] += 1
		if median:
			currentSet.sort()
			tmp = currentSet[len(currentSet) / 2]
			max_mid = train_data[tmp][index]
		else:
			midVec = self.getMidVec(valueSet)
			ll = len(midVec)
			for i in range(ll):

				mid = midVec[i]
				t = self.numeric_info(vec, mid, currentSet)
				if max_entropy < t:
					max_entropy = t
					max_mid = mid

		return [max_entropy, max_mid]

	def getEachEntropy(self, index, row):
		#call by nominal_info to calculate the entropy of each branch of nominal feature
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
		#call by helper_nominal to calculate
		acc = [[] for i in range( len(nominal[index]) )]
		count = 0
		l = len(vec)
		for count in range(l):
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
		#calculate the entropy of each feature
		vec = []
		for instance_index in currentSet:
			vec.append(train_data[instance_index][index])
		l = len(currentSet)
		entropy = self.nominal_info(vec, index, currentSet)
		return entropy
	


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def load_arff(data_name):
	global map_label
	data = {}
	data["attributes"] = []
	data["data"] = []
	enterData = 0
	with open(data_name) as file:
		count = 0
		for line in file:
			if enterData:
				row = []
				vec = re.split("[, \n\r]+", line)
				l = len(vec)
				for i in range(l):
					if isfloat(vec[i]) and vec[i] != "":
						row.append(float(vec[i]))
					elif vec[i] != "":
						row.append(vec[i])
				data["data"].append(row)
				continue
			vec = re.split("[ ,}\n\r']+", line)
			if len(vec) > 0 and vec[0] == "@attribute":
				if vec[1] == "class":
					map_label[vec[3]] = 0
					map_label[vec[4]] = 1

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

def printTree(node, layer):
	#print the tree
	if node.isNominal:
		l = len(node.child)
		base = ""
		for i in range(layer):
			base += splitter
		for i in range(l):
			s = ""
			s += base
			s += originAttributes[node.index]
			s += " = "
			s += str(node.threshold[i])
			s += " [" + str(node.child[i].n) + " " + str(node.child[i].p) + "]" 
			if node.child[i].isLeaf:
				s += ": " + inverseMap[node.child[i].prediction]
				print s
			else:
				print s
				printTree(node.child[i], layer + 1)

	else:
		s = ""
		for i in range(layer):
			s += splitter
		s += originAttributes[node.index]
		s += " <= "
		tmp = str(node.threshold)
		vv = tmp.split('.')
		if len(vv[1]) < 6:
			diff = 6 - len(vv[1])
			for i in range(diff):
				vv[1] += "0"
		tmp = vv[0] + "." + vv[1]
		s += tmp

		s += " [" + str(node.child[0].n) + " " + str(node.child[0].p) + "]" 
		if node.child[0].isLeaf:
			s += ": " + inverseMap[node.child[0].prediction]
			print s
		else:
			print s
			printTree(node.child[0], layer + 1)

		s = ""
		for i in range(layer):
			s += splitter
		s += originAttributes[node.index]
		s += " > "
		tmp = str(node.threshold)
		vv = tmp.split('.')
		if len(vv[1]) < 6:
			diff = 6 - len(vv[1])
			for i in range(diff):
				vv[1] += "0"
		tmp = vv[0] + "." + vv[1]
		s += tmp

		s += " [" + str(node.child[1].n) + " " + str(node.child[1].p) + "]" 

		if node.child[1].isLeaf:
			s += ": " + inverseMap[node.child[1].prediction]
			print s
		else:
			print s
			printTree(node.child[1], layer + 1)

	

def main():
	#dt-learn.py 
	global train_data, test_data, attributeNum, instanceNum, nominal, label, thresholdMap, thresholdNum
	global trainSetSize, stopThreshold, originAttributes, inverseMap, inverseThresholdMap
	argv = sys.argv
	argvNum = len(argv)
	if argvNum != 4:
		print "error arguments number"
		return
	train_data_name = argv[1]
	test_data_name = argv[2]
	stopThreshold = int(argv[3])

	train_data = load_arff(train_data_name)
	test_data = load_arff(test_data_name)

	#store feature name in originattributes
	originAttributes = []
	for ele in train_data["attributes"]:
		originAttributes.append(ele[0])

	#get the number of features
	attributeNum = len(train_data["attributes"]) - 1
	if attributeNum < 0:
		return

	#thresholdmap store the map from value to int for each feature
	thresholdMap = [{} for i in range(attributeNum)]
	inverseThresholdMap =  [{} for i in range(attributeNum)]

	#nominal stores the values of each features
	count = 0
	cur_row = train_data["data"][0]
	for i in range(attributeNum):
		ele = cur_row[i]
		if not type(ele) == str:
			nominal[count] = "NUMERIC"
		else:
			nominal[count] = []
			for instance in train_data["attributes"][count][1]:
				if nominal[count].count(instance) == 0:
					nominal[count].append(instance)
		count += 1

	#map the value of each feature to the index, in order to be used by the tree node
	for ele in nominal:
		count = 0
		if nominal[ele] != "NUMERIC":
			for item in nominal[ele]:
				thresholdMap[ele][item] = count
				inverseThresholdMap[ele][count] = item
				count += 1
	count = 0

	#map label to 0 and 1
	inverseMap = [0, 0]
	for ele in map_label:
		inverseMap[map_label[ele]] = ele
	# map_label["positive"] = 1
	# map_label["negative"] = 0

	# print map_label

	#store label of each instance
	for ele in train_data["data"]:
		label.append(map_label[ele[attributeNum]])

	#store the number of possible values of each feature
	thresholdNum = [0 for i in range(attributeNum)]
	for i in range(attributeNum):
		if not type(train_data["data"][0][i]) is str:
			continue
		thresholdNum[i] = len(nominal[i])
			
	visited = []
	instanceNum = len(train_data["data"])
	currentSet = [i for i in range(instanceNum)]



	train_data = train_data["data"]
	test_data = test_data["data"]

	root = node(visited, currentSet, 0)



	printTree(root, 0)

	print "<Predictions for the Test Set Instances>"
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
		print str(i + 1) + ": Actual: " + str(inverseMap[test_label[i]])  +" Predicted: "  +  str(inverseMap[predict_label[i]])
	# correct += 0.0
	print "Number of correctly classified: " + str(correct) +" Total number of test instances: "  + str(test_num)



if __name__ == "__main__":
	main()