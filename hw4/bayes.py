import re
import sys
import string
import math
from scipy.io import arff

TEST = 1
train_data = []
test_data = []
train_meta = []
test_meta = []
mat = {}
feature2index = {}
index2feature = [] #index to feature index
feature_num = 0
each_feature_value_num = {}
feature_values = {} #possible value in each feature
label_count = {}
total_count = 0


def pseducount(occurances, feature_vector):
	bias = len(feature_vector)
	subsum = sum(feature_vector) + bias + 0.0
	return (occurances + 1.0) / subsum
def naive_bayes():
	global label_count, total_count


	for instance in train_data:
		if instance[feature_num] not in label_count:
			label_count[instance[feature_num]] = 0
		label_count[instance[feature_num]] += 1
		total_count += 1
		for index in range(feature_num):
			label = instance[feature_num]
			mat[label][index][instance[index]] += 1

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
			pxy = pseducount( mat[label][index][instance[index]], mat[label][index].values() )
			py *= pxy
			
			tmp = pseducount( mat[other_label][index][instance[index]], mat[other_label][index].values() )
			other *= tmp
		num = other + py
		if other > py:
			prediction = other_label
			num = other / num
		else:
			prediction = label
			num = py / num
		print prediction + " " + label + " " + str(num)

def TAN():
	pass	
def main():
	global feature2index, index2feature, feature_values, feature_num
	global mat, train_meta, train_data, test_data, test_meta
	train_file = "lymph_train.arff"
	test_file = "lymph_test.arff"
	useTree = 'n'
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