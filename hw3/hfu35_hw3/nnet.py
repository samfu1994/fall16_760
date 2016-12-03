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
TEST = 0
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

	#build input layer vector, based on 1 of k policy
	for i in range(feature_num):
		cur = instance[i]
		feature_name = index2feature[i]

		if isNominal[feature_name]:
			cur_feature_len = each_feature_value_num[feature_name]
			cur_index = feature_values[feature_name].index(cur)
			tmp_vec = [0.0 for i in range(cur_feature_len)]
			tmp_vec[cur_index] = 1.0
			input_vec.extend(tmp_vec)

		else:
			input_vec.append(float(instance[i]))

	if len(input_vec) != input_unit_num:
		# print "not match input dimension"
		return

	input_vec.append(1.0) #bias

	#build hidden_vec
	hidden_vec_input = [0.0 for i in range(h)]
	hidden_vec_output = [0.0 for i in range(h)]
	for i in range(input_unit_num + 1):
		for j in range(h):
			hidden_vec_input[j] += w1[i][j] * input_vec[i]

	#implement sigmoid to get result of each hidden unit
	for i in range(h):
		hidden_vec_output[i] = sigmoid(hidden_vec_input[i])

	hidden_vec_output.append(1.0) #bias


	output_in = 0
	#get the input of output node
	for i in range(h + 1):
		output_in += hidden_vec_output[i] * w2[i] 
	output_out = sigmoid(output_in)

	#get the label 
	label = class2index[instance[feature_num]]

	cross_entropy = - label * np.log(output_out) - (1 - label) * np.log(1 - output_out)
	delta_cross_entropy = (output_out - label)

	#if prediction, return, do not modify weight matrix
	if isPredict == 1:
		prediction = -1
		if output_out > 0.5:
			prediction = 1
		else:
			prediction = 0
		# return prediction
		if prediction == label:
			return cross_entropy, 1
		else:
			return cross_entropy, 0

	elif isPredict == 2:
		prediction = -1
		if output_out > 0.5:
			prediction = 1
		else:
			prediction = 0
		# return prediction
		return output_out, prediction, label

	#back propagation
	delta_hidden = [0.0 for i in range(h + 1)]
	delta_w2 = [0.0 for i in range(h + 1)]
	#update weight    hidden -> output
	for i in range(h + 1):
		delta_hidden[i] = delta_cross_entropy 
		delta_w2[i] = delta_hidden[i] * hidden_vec_output[i]

	#update weight   input -> hidden
	for j in range(h):
		deriv_hidden = (1 - hidden_vec_output[j]) * hidden_vec_output[j]
		for i in range(input_unit_num + 1):
			w1[i][j] -= l * delta_hidden[j] * deriv_hidden * input_vec[i] * w2[j]

	for i in range(h + 1):
		w2[i] -= l * delta_w2[i]

	return cross_entropy


def train_direct_link(instance, w1, isPredict): 
	#for no hidden layer
	input_vec = []

	#build input layer vector
	for i in range(feature_num):
		cur = instance[i]
		feature_name = index2feature[i]

		if isNominal[feature_name]:
			cur_feature_len = each_feature_value_num[feature_name]
			cur_index = feature_values[feature_name].index(cur)
			tmp_vec = [0.0 for i in range(cur_feature_len)]
			tmp_vec[cur_index] = 1.0
			input_vec.extend(tmp_vec)

		else:
			input_vec.append(float(instance[i]))

	if len(input_vec) != input_unit_num:
		# print "not match input dimension"
		return

	input_vec.append(1.0) #bias

	output_in = 0
	for i in range(input_unit_num + 1):
		output_in += input_vec[i] * w1[i] 

	output_out = sigmoid(output_in)
	label = class2index[instance[feature_num]]

	cross_entropy = - label * np.log(output_out) - (1 - label) * np.log(1 - output_out)
	delta_cross_entropy = (output_out - label)


	if isPredict == 1:
		prediction = -1
		if output_out > 0.5:
			prediction = 1
		else:
			prediction = 0
		# return prediction
		if prediction == label:
			return cross_entropy, 1
		else:
			return cross_entropy, 0

	elif isPredict == 2:
		prediction = -1
		if output_out > 0.5:
			prediction = 1
		else:
			prediction = 0
		# return prediction
		return output_out, prediction, label

	#back propagation
	for i in range(input_unit_num + 1):
		w1[i] -= l * delta_cross_entropy * input_vec[i]

	return cross_entropy


def main():
	global feature2index, index2feature, isNominal, feature_values, feature_num, input_unit_num
	global l, h , e
	l = 0.0001
	h = 10
	e = 50
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
	# np.random.shuffle(test_data)
	

	i = 0
	#store the metadata
	for ele in train_meta:
		if ele == "class":
			for i in range(2):
				class2index[train_meta[ele][1][i]] = i
				index2class.append(train_meta[ele][1][i])
			continue
		feature2index[ele] = i    # feature name to index
		index2feature.append(ele)   #index to feature name
		isNominal[ele] = train_meta[ele][0] == 'nominal'  #key: feature name, value: is a nominal feature or not
		if isNominal[ele]:
			feature_values[ele] = train_meta[ele][1]  #if nominal feature, then store all the possible value of this feature
		else:
			feature_values[ele] = ()
		each_feature_value_num[ele] = len(feature_values[ele]) 
		i += 1

	feature_num = len(index2feature)

	for i in range(feature_num):
		if not isNominal[index2feature[i]]:
			column_vec = [0 for j in range(len(train_data))]
			for j in range(len(train_data)):
				column_vec[j] = train_data[j][i] 
			mysum = sum(column_vec)
			ave = mysum / (len(train_data) + 0.0)
			square_error = 0
			tmp = 0
			for j in range(len(train_data)):
				tmp = train_data[j][i] - ave
				square_error += tmp * tmp
			square_error /= (len(train_data) + 0.0)
			abs_error = math.sqrt(square_error)
			for j in range(len(train_data)):
				train_data[j][i] = (column_vec[j] - ave) / abs_error
			for j in range(len(test_data)):
				test_data[j][i] = (test_data[j][i] - ave) / abs_error
	print test_data


	# for i in range(feature_num):
	# 	if not isNominal[index2feature[i]]:


	#calculate the input vector length based on the possible value of each feature
	for ele in each_feature_value_num:
		if each_feature_value_num[ele] != 0:
			input_unit_num += each_feature_value_num[ele]
		else:
			input_unit_num += 1


	if h == 0:
		#no hidden layer
		w1 = [myrand() for i in range(input_unit_num + 1)]
		for i in range(e):
			error = 0
			err = 0
			correct = 0
			for instance in train_data:
				train_direct_link(instance, w1, 0)
			for instance in train_data:
				tmp_err, tmp_correct = train_direct_link (instance, w1, 1)
				err += tmp_err
				correct += tmp_correct
			# print str(i + 1) + "\t" + str("{0:.6f}".format(err)) + "\t" + str(correct) + "\t" + str(len(train_data) - correct)
			print ("%s\t%s\t%s\t%s" % (str(i + 1), str(err), str(correct), str(len(train_data) - correct) ) )

		correct = 0
		for instance in test_data:
			tmp_activation, tmp_predict, tmp_actual = train_direct_link(instance, w1, 2)
			correct += (tmp_predict == tmp_actual)
			# print str("{0:.6f}".format(tmp_activation)) + "\t" + str(tmp_predict) + "\t" + str(tmp_actual) 
			print ("%s\t%s\t%s" % (str(tmp_activation) , str(tmp_predict) , str(tmp_actual) ) )
		print ("%s\t%s" % (str(correct) , str(len(test_data) - correct)))

		
	else:
		#has hidden layer
		w1 = [[myrand() for i in range(h)] for i in range(input_unit_num + 1)]
		#(input + 1) * hidden,  input layer -> hidden layer

		w2 = [myrand() for i in range(h + 1)]   
		#(hidden + 1) *1,    hidden layer -> output

		for i in range(e):
			cross_entropy_error = 0
			err = 0
			correct = 0
			for instance in train_data:
				train(instance, w1, w2, 0)

			for instance in train_data:
				tmp_err, tmp_correct = train(instance, w1, w2, 1)
				err += tmp_err
				correct += tmp_correct
			# print str(i + 1) + "\t" + str("{0:.6f}".format(err)) + "\t" + str(correct) + "\t" + str(len(train_data) - correct)
			print ("%s\t%s\t%s\t%s" % (str(i + 1), str(err), str(correct), str(len(train_data) - correct) ) )


		correct = 0
		for instance in test_data:
			tmp_activation, tmp_predict, tmp_actual = train(instance, w1, w2, 2)
			correct += (tmp_predict == tmp_actual)
			# print str("{0:.6f}".format(tmp_activation)) + "\t" + str(tmp_predict) + "\t" + str(tmp_actual) 
			print ("%s\t%s\t%s" % (str(tmp_activation) , str(tmp_predict) , str(tmp_actual) ) )
		print ("%s\t%s" % (str(correct) , str(len(test_data) - correct)))







if __name__ == "__main__":
	main()