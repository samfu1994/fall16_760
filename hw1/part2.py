import math
import csv
import Queue
import sys
import copy
import re
import random
import pydot
from sets import Set
import collections
import matplotlib.pyplot as plt

train_data = [];
test_data = [];
label =[];
thresholdNum = []
attributeNum = 0;
instanceNum = 0;
nominal = {}
stopThreshold = 4;
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
		# print "current set " + str(len(currentSet))
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
					# nominal
			self.isNominal = 1
		else:
			self.isNominal = 0


		# self.visited.append(self.index);

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

				# print "		feature is : " + originAttributes[attribute_index]
				# print "		inforgain is : " + str(val)
				# print "		mid is : " + str(mid)

				#val is negative, plogp is smaller than 0 because p is smaller than 1
				if val > max_val + 0.000000000000001:
					max_val = val
					index = attribute_index
					max_mid = mid
			# print "				select : " + str(originAttributes[index])
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
	def getMidVec(self, valueSet):
		l = len(valueSet)
		s = Set()
		res = []
		pre_neg = -1
		pre_pos = -1
		for ele in valueSet:
			if ele not in s:
				s.add(ele)
		s = sorted(s, key = lambda ele : ele[0])
		for i in range(len(s)):
			cur = s[i][0]
			cur_label = s[i][1]
			for j in range(i + 1, len(s)):
				if cur == s[j][0]:
					continue
				if cur_label == s[j][1]:
					if j < len(s) - 1 and s[j + 1][0] == s[j][0] and s[j + 1][1] != s[j][1]:
						res.append((cur + s[j][0]) / 2)
					break
				res.append((cur + s[j][0]) / 2)

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
			valueSet.append((train_data[ele][index], label[ele]))
		valueSet.sort(key = lambda tup: tup[0]);

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
		#call by nominal_info to calculate
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
	data = {}
	data["attributes"] = []
	data["data"] = []
	enterData = 0
	with open(data_name) as file:
		count = 0
		for line in file:
			if enterData:
				row = []
				vec = re.split("[,\n ]+", line)
				l = len(vec)
				for i in range(l):
					if isfloat(vec[i]) and vec[i] != "":
						row.append(float(vec[i]))
					elif vec[i] != "":
						row.append(vec[i])
				data["data"].append(row)
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

def printTree(node, layer):

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
			s += "  [" + str(node.child[i].n) + " " + str(node.child[i].p) + "]" 
			if node.child[i].isLeaf:
				s += " : " + inverseMap[node.child[i].prediction]
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

		s += "  [" + str(node.child[0].n) + " " + str(node.child[0].p) + "]" 
		if node.child[0].isLeaf:
			s += " : " + inverseMap[node.child[0].prediction]
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

		s += "  [" + str(node.child[1].n) + " " + str(node.child[1].p) + "]" 

		if node.child[1].isLeaf:
			s += " : " + inverseMap[node.child[1].prediction]
			print s
		else:
			print s
			printTree(node.child[1], layer + 1)

	

def func(local_train_data, local_test_data, trainSetSize):
	#dt-learn.py 
	global train_data, test_data, attributeNum, instanceNum, nominal, label, thresholdMap, thresholdNum
	global stopThreshold, originAttributes, inverseMap, inverseThresholdMap
	accuracy = []
	copy_train_data = copy.deepcopy(local_train_data)
	copy_test_data = copy.deepcopy(local_test_data)
	if 1:
		originAttributes = []
		for ele in local_train_data["attributes"]:
			originAttributes.append(ele[0])

		attributeNum = len(local_train_data["attributes"]) - 1
		if attributeNum < 0:
			return

		#thresholdmap store the map from value to int for each feature
		thresholdMap = [{} for i in range(attributeNum)]
		inverseThresholdMap =  [{} for i in range(attributeNum)]

		#nominal stores the values of each features
		nominal = {}
		count = 0
		cur_row = local_train_data["data"][0]
		for i in range(attributeNum):
			ele = cur_row[i]
			if not type(ele) == str:
				nominal[count] = "NUMERIC"
			else:
				nominal[count] = []
				#local_train_data["attribuets"] is a tuple, two element, first name ,second list containing values
				for instance in local_train_data["attributes"][count][1]:
					if nominal[count].count(instance) == 0:
						nominal[count].append(instance)
			count += 1

		for ele in nominal:
			count = 0
			if nominal[ele] != "NUMERIC":
				for item in nominal[ele]:
					thresholdMap[ele][item] = count
					inverseThresholdMap[ele][count] = item
					count += 1
		map_label = {}
		count = 0

		#map label to 0 and 1
		inverseMap = [0, 0]
		inverseMap[0] = "negative"
		inverseMap[1] = "positive"
		map_label["positive"] = 1
		map_label["negative"] = 0



		thresholdNum = [0 for i in range(attributeNum)]
		for i in range(attributeNum):
			if not type(local_train_data["data"][0][i]) is str:
				continue
			thresholdNum[i] = len(nominal[i])
				
		visited = []

		candidate_train_data = local_train_data["data"]
		test_data = local_test_data["data"]
		train_length = len(candidate_train_data)

	for train_time in range(10):

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

		label = []
		for ele in train_data:
			label.append(map_label[ele[attributeNum]])

		instanceNum = len(train_data)
		currentSet = [i for i in range(instanceNum)]

		root = node(visited, currentSet, 0)

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
			# print str(i + 1) + ": Actual: " + str(inverseMap[test_label[i]])  +" Predicted: "  +  str(inverseMap[predict_label[i]])
		# correct += 0.0
		accuracy.append((correct + 0.0) / test_num)
		if trainSetSize == 1:
			break
	maxacc = 0
	sumacc = 0
	minacc = 1
	for ele in accuracy:
		sumacc += ele
		if ele > maxacc:
			maxacc = ele
		if ele < minacc:
			minacc = ele
	avgacc = sumacc / float(len(accuracy))
	return [avgacc, minacc, maxacc]

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

	sizeVec = [0.05, 0.1, 0.2, 0.5, 1]

	# train_data = load_arff('data/diabetes_train.arff')
	# test_data = load_arff('data/diabetes_test.arff')

	train_data = load_arff('data/heart_train.arff')
	test_data = load_arff('data/heart_test.arff')
	avgAcc = []
	minAcc = []
	maxAcc = []

	for ele in sizeVec:
		# stopThreshold = ele
		tmp = func(train_data, test_data, ele)
		avgAcc.append(tmp[0])
		minAcc.append(tmp[1])
		maxAcc.append(tmp[2])
	plt.subplot(111)
	print avgAcc
	print minAcc
	print maxAcc
	plt.plot(sizeVec, avgAcc, label = "average")
	plt.plot(sizeVec, minAcc, label = "min")
	plt.plot(sizeVec, maxAcc, label = "max")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
	plt.axis([0, 1, 0, 1])
	plt.ylabel("accuracy")
	plt.xlabel("training set size")
	plt.show()



if __name__ == "__main__":
	main()