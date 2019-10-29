##################


#Roll No. - 17CS10061
#Name - Rajat Kumar Jenamani
#Assignment Number - 2
#Execution - $python2 [name of the file]


##################
import numpy as np
import pandas as pd
import csv

def read_data(filename):
	with open(filename, 'r') as f:
		reader = f.readlines()
		data_frame = list(csv.reader([string[1:-3] for string in reader]))
		data = pd.DataFrame(data_frame[1:], columns=data_frame[0])
		data[data.columns]=data[data.columns].astype(int)
	return data

def find_prob(data,class_num,col,val):
	data_match = data[data['D'] == class_num]
	true_match = data_match[data_match[col]==val]
	return (float(len(true_match)+1))/(len(data_match)+5)

def prob_matrix(data):
	P = []
	for class_num in range(0,2):
		Q=[]
		for col in data.columns[1:]:
			R=[]
			for val in range(1,6):
				R.append(find_prob(data,class_num,col,val))
			Q.append(R)
		P.append(Q)
	P_matrix = np.array(P)
	return P_matrix

def classify(P_matrix,data,attributes,zero_prob,one_prob):
	zero_class_P=1.0
	one_class_P=1.0
	for col in range(len(attributes)-1):
		zero_class_P = zero_class_P * P_matrix[0][col][attributes[col+1]-1]
		one_class_P = one_class_P * P_matrix[1][col][attributes[col+1]-1]
	if zero_class_P*zero_prob > one_class_P*one_prob:
		return 0
	else:
		return 1

def get_accuracy(data,P_matrix):
	zero_prob = len(data[data['D'] == 0])
	one_prob = len(data[data['D'] == 1])
	correct_prediction_count=0
	for i, values in data.iterrows():
		if (values.iloc[0] == classify(P_matrix,data,values,zero_prob,one_prob) ):	
			correct_prediction_count = correct_prediction_count + 1 
	return float(correct_prediction_count*100)/data.shape[0]

if __name__ == '__main__':

	train_data = read_data('data2_19.csv')
	P_matrix = prob_matrix(train_data)
	# print(P_matrix)
	print('Training Set Accuracy: ',get_accuracy(train_data,P_matrix))

	test_data = read_data('test2_19.csv')
	print('Test Set Accuracy: ',get_accuracy(test_data,P_matrix))