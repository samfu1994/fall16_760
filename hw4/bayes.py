import re
import sys
import string
import math
import numpy as np
from math import log
from scipy.io import arff

TEST = 0
train_data = []
test_data = []
train_meta = []
test_meta = []
mat = {}
feature2index = {}
index2feature = [] #index to feature index
index2label = []
label2index = {}
feature_num = 0
each_feature_value_num = {}
feature_values = {} #possible value in each feature
label_count = {}
total_count = 0
parent = {}
root_name = 0
count_0 = 0
count_1 = 0

prob_mat = {}

def pseducount(occurances, subsum, bias):
	return (occurances + 1.0) / (subsum + bias)
def naive_bayes():
	global label_count, total_count

	for feature in index2feature:
		print feature + " class"

	print ""
	for instance in train_data:
		if instance[feature_num] not in label_count:
			label_count[instance[feature_num]] = 0
		label_count[instance[feature_num]] += 1
		total_count += 1
		for index in range(feature_num):
			label = instance[feature_num]
			mat[label][index][instance[index]] += 1

	correct = 0
	for instance in test_data:
		prediction = 0
		num = 0
		label = instance[feature_num]
		other_label = 0
		for key in mat:
			if key == label:
				continue
			other_label = key
		py = (label_count[label] + 1.0) / (total_count + 2.0)
		other = (label_count[other_label] + 1.0) / (total_count + 2.0)
		for index in range(feature_num):
			pxy = pseducount( mat[label][index][instance[index]], sum(mat[label][index].values() ), len(mat[label][index]) )
			py *= pxy
			
			tmp = pseducount( mat[other_label][index][instance[index]], sum(mat[other_label][index].values()) , len(mat[other_label][index]))
			other *= tmp
		num = other + py
		if other > py:
			prediction = other_label
			num = other / num
		else:
			prediction = label
			num = py / num
		prediction = prediction.replace("\'", "")
		label = label.replace("\'", "")

		print prediction + " " + label + " " + str(num)
		correct += prediction == label

	print ""
	print correct
def countSet():
	global feature_num
	count_vec = [0, 0]
	for instance in train_data:
		label = instance[feature_num]
		count_vec[label2index[label]] += 1
	return count_vec[0], count_vec[1]

def getMutualInformation(feature_i, feature_j):
	global feature_values, index2feature, count_0, count_1
	mutualInfo = 0
	possibility_xi = 1
	possibility_xj = 1
	possibility_xi_xj = 1
	possibility_xi_xj_y = 1
	counter_xi = 0
	counter_xj = 0
	counter_xi_xj = 0
	counter_xi_xj_y = 0
	count_0, count_1 = countSet()
	for k in range(2):
		for j in range(len(feature_values[feature_j])):
			for i in range(len(feature_values[feature_i])):
				for instance in train_data:
					if instance[feature_i] == feature_values[feature_i][i] and\
					instance[-1] == index2label[k]:
						counter_xi += 1
					if instance[feature_j] == feature_values[feature_j][j] and\
					instance[-1] == index2label[k]:
						counter_xj += 1
					if instance[feature_i] == feature_values[feature_i][i]\
						and instance[feature_j] == feature_values[feature_j][j]\
						and instance[-1] == index2label[k]:
						counter_xi_xj += 1
					if instance[feature_i] == feature_values[feature_i][i] and\
					 instance[feature_j] == feature_values[feature_j][j] and\
					 instance[-1] == index2label[k]:
					 	counter_xi_xj_y += 1
				if k == 0:
					possibility_xi = (counter_xi + 1 + 0.0) / (count_0 + len(feature_values[feature_i]))
					possibility_xj = (counter_xj + 1 + 0.0) / (count_0 + len(feature_values[feature_j]))
					possibility_xi_xj = (counter_xi_xj + 1 + 0.0) / (count_0 + len(feature_values[feature_j]) * \
						len(feature_values[feature_i]))
				elif k == 1:
					possibility_xi = (counter_xi + 1 + 0.0) / (count_1 + len(feature_values[feature_i]))
					possibility_xj = (counter_xj + 1 + 0.0) / (count_1 + len(feature_values[feature_j]))
					possibility_xi_xj = (counter_xi_xj + 1+ 0.0) / (count_1 + len(feature_values[feature_j]) * \
						len(feature_values[feature_i]))
				possibility_xi_xj_y = (counter_xi_xj_y + 1 + 0.0) / (len(train_data) + len(feature_values[feature_j]) * \
						len(feature_values[feature_i]) * len(index2label))
				mutualInfo += possibility_xi_xj_y * log((possibility_xi_xj / (possibility_xi * possibility_xj)), 2)
				counter_xi = 0
				counter_xj = 0
				counter_xi_xj = 0
				counter_xi_xj_y = 0
	return mutualInfo

def prim():
	global index2feature, parent, root_name
	root_name = index2feature[0]
	remain = []
	parent[root_name] = root_name
	already_add = [root_name]
	for i in range(1, len(index2feature)):
		remain.append(index2feature[i])

	while len(remain) != 0:
		biggest = -np.inf
		index = -1
		cur_parent = -1
		for i in range(len(remain)):
			for j in range(len(already_add)):
				tmp = getMutualInformation(remain[i], already_add[j])
				if tmp > biggest:
					biggest = tmp
					index = i
					cur_parent = j
		parent[remain[index]] = already_add[cur_parent]
		already_add.append(remain[index])
		remain.pop(index)

	print index2feature[0] + " class"
	for i in range(1, len(index2feature)):
		print index2feature[i] + " " + parent[index2feature[i]] + " class"
	print ""

def TAN():
	global prob_mat, train_data, feature_num
	prim()
	# 4 dimension, label * name * value * parent_value
	for label in index2label:
		prob_mat[label] = {}
		for name in parent:
			prob_mat[label][name] = {}
			for feature_value in feature_values[name]:
				prob_mat[label][name][feature_value] = {}
				for parent_value in feature_values[ parent[name] ]:
					prob_mat[label][name][feature_value][parent_value] = 0


	for instance in train_data:
		label = instance[feature_num]
		for cur_feature_index in range(feature_num):
			cur_feature_val = instance[cur_feature_index]
			cur_feature_name = index2feature[cur_feature_index]
			parent_index = feature2index[ parent[cur_feature_name] ]
			parent_val = instance[parent_index]
			prob_mat[label][cur_feature_name][cur_feature_val][ parent_val ] += 1

	for i in range(2):
		label = index2label[i]
		subsum = 0
		for entry in prob_mat[label][root_name]:
			subsum += prob_mat[label][root_name][entry][entry]
		for feature_value in feature_values[root_name]:			
			prob_mat[label][root_name][feature_value][feature_value] = pseducount(prob_mat[label][root_name][feature_value][feature_value], subsum, len(feature_values[root_name]))


	for k in range(2):
		label = index2label[k]
		for i in range(1, feature_num):
			totalsum = 0
			feature_name = index2feature[i]
			parent_name = parent[feature_name]
			for parent_value in feature_values[parent_name]:
				subsum = 0
				for feature_value in feature_values[feature_name]:
					subsum += prob_mat[label][feature_name][feature_value][parent_value]
				
				for feature_value in feature_values[feature_name]:
					tmp = prob_mat[label][feature_name][feature_value][parent_value]
					prob_mat[label][feature_name][feature_value][parent_value] = pseducount(tmp, subsum, len(feature_values[feature_name] ) )
	correct = 0
	for instance in test_data:
		divisor = 0
		pc = [0.0, 0.0]
		PRootC = [0.0, 0.0]
		label = instance[feature_num]
		for train_instance in train_data:
			if list(train_instance)[:-1] == list(instance)[:-1]:
				divisor += 1

		bias = 1
		for entry in feature_values:
			bias *= len(feature_values[entry])

		divisor = pseducount(divisor, count_0 + count_1, bias)
		pc[0] = pseducount(count_0, count_0 + count_1, 2)
		pc[1] = pseducount(count_1, count_0 + count_1, 2)

		subsum = 0
		for i in range(2):
			label = index2label[i]
			PRootC[i] = prob_mat[label][root_name][instance[0]][instance[0]]

		for count in range(1, feature_num):
			cur = index2feature[count]
			p = parent[cur]
			for i in range(2):
				label = index2label[i]
				parent_index = feature2index[p]
				PRootC[i] *= prob_mat[label][cur][instance[count]][instance[parent_index]]

		prob = [0.0, 0.0]
		prob[0] = PRootC[0] * pc[0] / divisor
		prob[1] = PRootC[1] * pc[1] / divisor
		prediction = 0
		num = 0
		if prob[0] > prob[1]:
			prediction = index2label[0]
			num = prob[0]
		else:
			prediction = index2label[1]
			num = prob[1]

		prediction = prediction.replace("\'", "")
		instance[-1] = instance[-1].replace("\'", "")
		print prediction + " " + instance[-1] + " " + str(num / (prob[0] + prob[1]))

		correct += prediction == instance[-1]
	print ""
	print correct
def main():
	global feature2index, index2feature, feature_values, feature_num
	global mat, train_meta, train_data, test_data, test_meta, label2index, index2label
	train_file = "lymph_train.arff"
	test_file = "lymph_test.arff"
	useTree = 't'
	if not TEST:
		argv = sys.argv
		argvNum = len(argv)
		train_file = argv[1]
		test_file = argv[2]
		useTree = argv[3]

	train_data, train_meta = arff.loadarff(train_file)
	test_data, test_meta = arff.loadarff(test_file)

	i = 0
	for ele in train_meta:
		if ele == "class":
			mat[train_meta[ele][1][0]] = []
			mat[train_meta[ele][1][1]] = []
			index2label = [train_meta[ele][1][0], train_meta[ele][1][1]]
			label2index[train_meta[ele][1][0]] = 0
			label2index[train_meta[ele][1][1]] = 1
			continue
		feature2index[ele] = i    # feature name to index
		index2feature.append(ele)   #index to feature name
		feature_values[ele] = train_meta[ele][1]  #if nominal feature, then store all the possible value of this feature
		each_feature_value_num[ele] = len(feature_values[ele])
		i += 1
	feature_num = len(index2feature)
	for entry in mat:
		mat[entry] = [{} for i in range(feature_num)]
	for feature_name in train_meta:
		if feature_name == "class":
			break
		feature_values_vector = train_meta[feature_name][1]
		for class_name in mat:
			for value in feature_values_vector:
				mat[class_name][feature2index[feature_name]][value] = 0

	if useTree == 'n':
		naive_bayes()
	else:
		TAN()



if __name__ == "__main__":
	main()