##################

#Roll No. - 17CS10061
#Name - Rajat Kumar Jenamani
#Assignment Number - 3
#Execution - $python2 [name of the file]

##################

from __future__ import print_function
import csv
import math
import numpy as np

####################### NODES OF DECISION TREE

class Leaf_Node:        # leaf node of the decision tree

	def __init__(self,data):
		self.prediction = find_class(data)  # prediction of the leaf node

class Internal_Node:    # the non leaf nodes of decision tree

	def __init__(self,
				 attribute,
				 branches):
		self.attribute = attribute # the attribute the data is split on
		self.branches = branches    # the branches of this node

########################

def unique_elements(input_list):  # return list of unique elements in input_list
	unique_list = []  
	for x in input_list: 
		if x not in unique_list: 
			unique_list.append(x) 
	return unique_list

def find_class(data):  #find class for given data
	number_of_features = len(data[0]) - 1  # number of features in data
	numPos=0  #number of data points where survived = yes
	numNeg=0  #number of data points where survived = no
	for x in data:
		if x[number_of_features] == 'yes':
			numPos=numPos+1
		else:
			numNeg=numNeg+1 
	if numPos>numNeg: # if number of datapoints where survived = yes > survived = no return 'yes'
		return 'yes' 
	else:
		return 'no'   

def find_entropy(data):
	n = len(data)
	# print(n)
	number_of_features = len(data[0]) - 1  # number of features in data
	numPos=0   #number of data points where survived = yes
	numNeg=0   #number of data points where survived = no
	for x in data:
		if x[number_of_features] == 'yes':
			numPos=numPos+1
		else:
			numNeg=numNeg+1
	if numPos == 0 or numNeg == 0:
		return 0.0
	# print('numPos',numPos,'numNeg',numNeg)
	probPos = float(numPos)/n   # probability of survived = yes
	probNeg = float(numNeg)/n   # probability of survived = no

	# print('probPos',probPos,' probNeg',probNeg)

	return -probPos * math.log(probPos,2) - probNeg * math.log(probNeg,2)  #entropy

def select_subset(data, attribute, value): # selects subset of data which has attribute value = value
	subset = []
	for x in data:
		if x[attribute] == value:  # if value of attribute is 'value' add data_point to subset
			subset.append(x) 
	return subset

def find_gain(data, attribute, values): # finds gain when data is split by attribute
	weighted_entropy = 0.0
	for v in values:
		subset = select_subset(data, attribute, v)
		weighted_entropy += find_entropy(subset) * len(subset)
	return find_entropy(data) - weighted_entropy/len(data)  #information gain

def best_split(data):
	best_attribute = None  # the best attribute 
	best_gain = 0  # the best information gain
	n_features = len(data[0]) - 1  # number of columns
	for col in range(n_features):  # for each feature

		values = unique_elements(list([row[col] for row in data]))  # unique values in the column

		if len(values)==1:  # if no unique values then continue
			continue

		gain = find_gain(data, col, values)   # find gain for given feature

		# print(' col:',col,' val:',values,' gain:',gain)
		if gain >= best_gain:               # store the best gain and best attribute
			best_attribute = col
			best_gain = gain 

	return best_attribute, best_gain  # return best attribute and best gain

def display_tree(node, spacing=""):
	# print(node)
	if isinstance(node, Leaf_Node):     # Base case: reached a leaf
		print (':',node.prediction,end =" ")
	else:              # Call this function recursively on the branchs of tree
		for branch in node.branches:
			print('\n',spacing + '----> ',headings[branch[1]],'=',branch[2],end =" ") #print the attribute: value
			display_tree(branch[0], spacing + "    ")
		# print(node.branches)

def create_tree(data):

	attribute, gain = best_split(data)

	# print('best split question:',question,'gain: ',gain)

	if gain == 0: #Base Case
		return Leaf_Node(data) 

	values = unique_elements(list([row[attribute] for row in data]))   # all unique values for given attribute

	branches=[]
	for val in values:
		subset_rows = select_subset(data,attribute,val)  # take subset of data where value of attribute = val
		branch = create_tree(subset_rows)                   # call build tree for subset
		branches.append( (branch , attribute, val) )        # append the child to brances of current node
	# print('branches: ',branches)
	return Internal_Node(attribute, branches)

def classify(node,test_data_point):
	if isinstance(node, Leaf_Node):     # Base case: reached a leaf
		return node.prediction          #return prediction

	for branch in node.branches:
		if test_data_point[branch[1]] == branch[2]:     # recurse for branch using value of attribute of test_data_point
			return classify(branch[0], test_data_point)

if __name__ == '__main__':

	# read data 
	with open('data3_19.csv', 'r') as f:
		reader = csv.reader(f)
		data = list(reader)

	headings = data[0] #store headings
	data.pop(0) #remove headings from list
	# print(data)

	data_size = int(len(data)*0.5)
	# print(data_size)

	# targets = data[-1]
	weights = np.full(len(data), 1.0/len(data))
	# print(weights)

	decision_trees = []
	decision_trees_weights = []
	indices = np.arange(len(data))

	iterations = 3

	for i in range(iterations):
		
		sample_data_indices = np.random.choice(indices, data_size, p=weights)
		# sample_data_indices = np.sort(sample_data_indices)
		# sample_data_indices = np.random.choice(indices, data_size, p=weights, replace = False)
		sample_data = []

		# print("Sample Size:",len(sample_data_indices))
		# print(sample_data_indices)

		for j in sample_data_indices:
			sample_data.append(data[j])

		decision_tree = create_tree(sample_data)   #build the tree
		# print('\nDECISION TREE:')
		# display_tree(decision_tree)           #print the tree
		
		count = 0     #count of correct prediction
		for row in sample_data:
			prediction = classify(decision_tree,row)
			# print(pred)
			if prediction == row[len(row)-1]:       #increase count if prediction matches true val
				count = count + 1       
				# print(count)
		# print(count)
		# print('Local Accuracy:', count * 100.0 /len(sample_data))  #prints accuracy

		
		epsilon = ((len(sample_data)-count) * 1.0) /len(sample_data)
		# print("epsilon:",epsilon)
		alpha = 0.5*math.log((1 - epsilon) / ( 1e-5 + epsilon))
		decision_trees_weights.append(alpha)

		# print("Alpha:",alpha)

		# update the weights
		for j in sample_data_indices:
			row = data[j]
			prediction = classify(decision_tree,row)
			# print(pred)
			if prediction == row[len(row)-1]:  
				weights[j] = math.exp(-1*alpha) * weights[j]
			else:
				weights[j] = math.exp(alpha) * weights[j]
		
		# print("OLD WEIGHTS:")
		# print(weights)
		normalization_factor = np.sum(weights,dtype=float)
		# print("Sum of Weights",sum_weights)
		weights = weights / normalization_factor
		# print("WEIGHTS:")
		# print(weights)

		# save the classifier
		decision_trees.append(decision_tree)

	# read data 
	with open('test3_19.csv', 'r') as f:
		reader1 = csv.reader(f)
		test_data = list(reader1)

	# print(test_data)

	count = 0     #count of correct prediction
	for row in test_data:
		pred = 0
		for i in range(iterations):
			prediction = classify(decision_trees[i],row)
			# print(pred)
			if prediction == row[len(row)-1]:       #increase count if prediction matches true val
				pred = pred + decision_trees_weights[i]       
			else:
				pred = pred - decision_trees_weights[i]

		if pred >= 0:
			count = count+1

	# print(count)
	# print(len(test_data))
	print('\nAccuracy:',(count * 100.0) /len(test_data))  #prints accuracy