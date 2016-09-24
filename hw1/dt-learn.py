import arff
import math
import csv
import Queue


train_data = [];
test_data = [];
label =[];
thresholdNum = []
attributeNum = 0;
instanceNum = 0;
nominal = {}
stopThreshold = 2;
thresholdMap ={}
MAX_INIT = -2147483648
MIN_INIT = 2147483647
median = 1
def getPrediction(node):
	p = 0
	n = 0
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
		print ("")
		print (len(currentSet))
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
		print "gain is " + str(gain)
		#if no information gain, then return
		if gain <= 0:
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
			child = self.getChild(currentSet, self.index, self.threshold)
		return

	def predict(self, instance):
		if self.isLeaf:
			# print ("prediction is " + str(self.index))
			return self.prediction
		else:
			branch = self.predict_helper(instance)
			# print ("I am " + str(self.index) + "  go to " + str(branch))
			return self.child[branch].predict(instance)
	def predict_helper(self, instance):
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
				feature_belong_class = thresholdMap[index][rain_data[i][index]]
				childList[feature_belong_class].append(i)
			
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
				else:
					val = self.helper_nominal(attribute_index, currentSet[:])
				# print "		finish attribute " + str(attribute_index)
				#val is negative
				print "    " + str(attribute_index)
				print "            " + str(val)
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
			print ("select  " + str(index) )
			print ("threshold : " + str(mid))
			self.index = index
			self.threshold = mid
			return gain
	


	def judgeStop(self, visited, currentSet):
		#no more split candidate
		if len(visited) == attributeNum:
			return 1

		#have less than m instances
		if len(currentSet) < stopThreshold:
			print "have less than m instances"
			return 1

		#has only one class
		l = []
		for i in currentSet:
			if l.count(label[i]) != 0:
				l.append(label[i])
		if len(l) == 1:
			print "has only one class"
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
			entropy += cur_l / l * getEachEntropy(index, row)
		return entropy

	def helper_nominal(self, index, currentSet):
		vec = []
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
		s += str(tmp.index)
		s += "    "
		for i in range(l):
			qq.put(tmp.child[i])
	print "      " + s
	check(qq)

def main():
	global train_data, test_data, attributeNum, instanceNum, nominal, label, thresholdMap

	# train_data = arff.load(open('data/diabetes_train.arff'))
	# test_data = arff.load(open('data/diabetes_test.arff'))

	train_data = arff.load(open('data/heart_train.arff'))
	test_data = arff.load(open('data/heart_test.arff'))

	attributeNum = len(train_data["attributes"]) - 1
	if attributeNum < 0:
		return

	print train_data["attributes"]

	#nominal store the nominal attribute name and value 
	# count = 0
	# for ele in train_data["attributes"]:
	# 	if ele[0] != "class": 
	# 		nominal[count] = ele[1]
	# 		thresholdMap[count] = {}
	# 	count += 1

	count = 0
	cur_row = train_data["data"][0]
	for ele in cur_row:
		if str(ele).isdigit:
			nominal[count] = "NUMERIC"
		else:
			nominal[count] = []
			for instance in train_data["data"]:
				if nominal[count].count(instance[count]) == 0:
					nominal[count].append(instance[count])

		count += 1
	print "nominal is " 
	print "     " + str(nominal)
	for ele in nominal:
		count = 0
		if nominal[ele] != "NUMERIC":
			for item in nominal[ele]:
				thresholdMap[ele][item] = count
				count += 1

	map_label = {}
	count = 0

	#map label to 0 and 1
	for ele in train_data["data"]:
		if ele[attributeNum] not in map_label:
			map_label[ele[attributeNum]] = count
			count += 1
		label.append(map_label[ele[attributeNum]])
	print label
	thresholdNum = [0 for i in range(attributeNum)]
	for i in range(attributeNum):
		if str(train_data["data"][0][i]).isdigit():
			continue;
		tmp_list = []
		for row in train_data["data"]:
			tmp = row[i]
			if tmp_list.count(tmp) == 0:
				thresholdNum[i] += 1

			
	visited = []
	instanceNum = len(train_data["data"])
	currentSet = [i for i in range(instanceNum)]

	train_data = train_data["data"]
	with open("data/c.csv", 'wb') as csvfile:
		writer = csv.writer(csvfile);
		for i in train_data:
			writer.writerows([i])
	test_data = test_data["data"]
	root = node(visited, currentSet, 0)

	print ("-------------------------------------------------------------------------------------------")

	q = Queue.Queue()
	q.put(root)
	check(q)


	train_label = []
	predict_label = []

	for ele in train_data:
		train_label.append(map_label[ele[attributeNum]])
		predict_label.append(root.predict(ele))

	train_num = len(train_label)
	correct = 0
	for i in range(train_num):
		# print str(train_label[i]) + "    " +  str(predict_label[i])
		if train_label[i] == predict_label[i]:
			correct += 1
	correct += 0.0
	print ("correct rate:" + str(correct / train_num) )



	test_label = []
	predict_label = []
	for ele in test_data:
		test_label.append(map_label[ele[attributeNum]])
		predict_label.append(root.predict(ele))

	test_num = len(test_label)
	correct = 0
	for i in range(test_num):
		# print str(test_label[i]) + "    " +  str(predict_label[i])
		if test_label[i] == predict_label[i]:
			correct += 1
		print str(test_label[i]) +"      "  + str(predict_label[i])
	correct += 0.0
	print ("correct rate:" + str(correct / test_num) )


if __name__ == "__main__":
	main()