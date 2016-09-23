from sklearn import tree
import arff


train_data = arff.load(open('data/diabetes_train.arff'))
test_data = arff.load(open('data/diabetes_test.arff'))
attributeNum = len(train_data["attributes"]) - 1
data =[]
t_data = []
for ele in train_data["data"]:
	data.append(ele[:attributeNum])

for ele in test_data["data"]:
	t_data.append(ele[:attributeNum])

map_label = {}
label = []
count = 0
for ele in train_data["data"]:
	if ele[attributeNum] not in map_label:
		map_label[ele[attributeNum]] = count
		count += 1
	label.append(map_label[ele[attributeNum]])




clf = tree.DecisionTreeClassifier()
clf.fit(data, label)



map_label = {}
label = []
count = 0
for ele in test_data["data"]:
	if ele[attributeNum] not in map_label:
		map_label[ele[attributeNum]] = count
		count += 1
	label.append(map_label[ele[attributeNum]])



pre = clf.predict(t_data)


l = len(pre)
correct = 0.0
for i in range(l):
	if pre[i] == label[i]:
		correct += 1

print correct / l