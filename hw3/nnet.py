from scipy.io import arff
import sys
import numpy as np
import random
import math



feature2index = {}
index2feature = [] #index to feature index
class2index = {} 
index2class = [] # index to class 0 or 1
isNominal = {} #name to true or false
TEST = 1
feature_num = 0
each_feature_value_num = {}
feature_values = {} #possible value in each feature
input_unit_num = 0
l = 0
h = 0
e = 0


def myrand():
	a = random.randrange(-1000, 1000)
	a += 0.0
	a /= 100000

	return a

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def train(instance, w1, w2, isPredict):
	input_vec = []

	#build input layer vector
	for i in range(feature_num):
		cur = instance[i]
		feature_name = index2feature[i]

		if isNominal[feature_name]:
			cur_feature_len = each_feature_value_num[feature_name]
			cur_index = feature_values[feature_name].index(cur)
			tmp_vec = [0 for i in range(cur_feature_len)]
			tmp_vec[cur_index] = 1
			input_vec.extend(tmp_vec)

		else:
			input_vec.append(float(instance[i]))

	if len(input_vec) != input_unit_num:
		print "not match input dimension"
		return

	input_vec.append(1) #bias


	#build hidden_vec
	hidden_vec_input = [0 for i in range(h)]
	hidden_vec_output = [0 for i in range(h)]
	for i in range(input_unit_num + 1):
		for j in range(h):
			hidden_vec_input[j] += w1[i][j] * input_vec[i]

	for i in range(h):
		hidden_vec_output[i] = sigmoid(hidden_vec_input[i])

	hidden_vec_output.append(1)


	output_in = 0
	for i in range(h + 1):
		output_in += hidden_vec_output[i] * w2[i] 

	output_out = sigmoid(output_in)
	label = class2index[instance[feature_num]]
	diff = label - output_out

	if isPredict:
		return diff

	#back propagation
	hidden_diff = [0 for i in range(h + 1)]
	for i in range(h + 1):
		hidden_diff[i] = diff * w2[i]



	#update weight   input -> hidden
	for j in range(h):
		deriv = (1 - sigmoid(hidden_vec_input[j])) * sigmoid(hidden_vec_input[j])
		for i in range(input_unit_num + 1):
			w1[i][j] += l * hidden_diff[j] * input_vec[i] * deriv 



	#update weight  hidden -> output
	deriv =  (1 - sigmoid(output_in)) * sigmoid(output_in)
	for i in range(h + 1):
		w2[i] += l * hidden_vec_output[i] * diff * deriv



	return diff

def train_direct_link(instance, w1, isPredict):
	input_vec = []

	#build input layer vector
	for i in range(feature_num):
		cur = instance[i]
		feature_name = index2feature[i]

		if isNominal[feature_name]:
			cur_feature_len = each_feature_value_num[feature_name]
			cur_index = feature_values[feature_name].index(cur)
			tmp_vec = [0 for i in range(cur_feature_len)]
			tmp_vec[cur_index] = 1
			input_vec.extend(tmp_vec)

		else:
			input_vec.append(float(instance[i]))

	if len(input_vec) != input_unit_num:
		print "not match input dimension"
		return

	input_vec.append(1) #bias
	#build hidden_vec


	output_in = 0
	for i in range(input_unit_num + 1):
		output_in += input_vec[i] * w1[i] 

	output_out = sigmoid(output_in)
	label = class2index[instance[feature_num]]
	diff = label - output_out

	if isPredict:
		return diff

	#back propagation


	#update weight  hidden -> output
	deriv =  (1 - sigmoid(output_in)) * sigmoid(output_in)
	for i in range(input_unit_num + 1):
		w1[i] += l * input_vec[i] * diff * deriv


	return diff


def main():
	global feature2index, index2feature, isNominal, feature_values, feature_num, input_unit_num
	global l, h , e
	l = 0.1
	h = 20
	e = 100
	train_file = "heart_train.arff"
	test_file = "heart_test.arff"
	if not TEST:
		argv = sys.argv
		argvNum = len(argv)
		l = float(argv[1])
		h = int(argv[2])
		e = int(argv[3])
		train_file = argv[4]
		test_file = argv[5]


	train_data, train_meta = arff.loadarff(train_file)
	test_data, test_meta = arff.loadarff(test_file)

	#first shuffle the mat
	np.random.shuffle(train_data)
	

	i = 0
	for ele in train_meta:
		if ele == "class":
			for i in range(2):
				class2index[train_meta[ele][1][i]] = i
				index2class.append(train_meta[ele][1][i])
			continue
		feature2index[ele] = i
		index2feature.append(ele)
		isNominal[ele] = train_meta[ele][0] == 'nominal'
		if isNominal[ele]:
			feature_values[ele] = train_meta[ele][1]
		else:
			feature_values[ele] = ()
		each_feature_value_num[ele] = len(feature_values[ele])
		i += 1

	feature_num = len(index2feature)

	
	for ele in each_feature_value_num:
		if each_feature_value_num[ele] != 0:
			input_unit_num += each_feature_value_num[ele]
		else:
			input_unit_num += 1


	if h == 0:
		w1 = [myrand() for i in range(input_unit_num + 1)]
		for i in range(e):
			error = 0
			for instance in train_data:
				error += train_direct_link(instance, w1, 0)
			print error

		for instance in test_data:
			error = train_direct_link(instance, w1, 1)
	else:
		w1 = [[myrand() for i in range(h)] for i in range(input_unit_num + 1)]
		#(input + 1) * hidden
		w2 = [myrand() for i in range(h + 1)]
		#(hidden + 1) *1

		for i in range(e):
			error = 0
			for instance in train_data:
				error += train(instance, w1, w2, 0)
			print error

		for instance in test_data:
			error = train(instance, w1, w2, 1)
		print error

		





if __name__ == "__main__":
	main()