##################

#Roll No. - 17CS10061
#Name - Rajat Kumar Jenamani
#Assignment Number - 4
#Execution - $python2 [name of the file]

##################

from __future__ import print_function
import csv
import math
import numpy as np
import random

def unique(input_list): 
		x = np.array(input_list) 
		return np.unique(x) 

def remove_labels(input_list):
	
	out_list = []
	for row in input_list:
		out_list.append(row[0:len(row)-1])

	for idx in range(len(out_list)):
		for idy in range(len(out_list[idx])):
			out_list[idx][idy] = float(out_list[idx][idy])

	return out_list

def euclid_distance(point1, point2):
	value = 0.0
	# print(point1)
	# print(point2)
	for i in range(len(point1)):
		value = value + (float(point1[i]) - float(point2[i]))** 2
	# print(math.sqrt(value))
	return math.sqrt(value)

def k_means(data,classes,iters = 10):
	random.seed(0)
	centroid_points = remove_labels(random.sample(data, k=len(classes)))
	dimensions = len(centroid_points[0])
	# print(dimensions)
	# print(centroid_points)

	# print(len(classes))

	clusters = [[] for x in range(len(classes))]

	for i in range(iters):
		
		
		clusters = [[] for x in range(len(classes))]
		# print(clusters)
		for idx in range(len(data)):
				
			distances = []
			for j in range(len(classes)):
				distances.append(euclid_distance(centroid_points[j],data[idx][0:len(data[idx])-1])) 

			class_index = distances.index(min(distances))

			clusters[class_index].append(data[idx][0:len(data[idx])-1])

		# print(clusters)

		for class_idx in range(len(classes)):

			mean = [0 for x in range(dimensions)]

			for member in clusters[class_idx]:
				# print(member)
				for idx in range(dimensions):
					mean[idx] = mean[idx] + float(member[idx])

			for idx in range(dimensions):
				centroid_points[class_idx][idx] = mean[idx]/len(clusters[class_idx])

			# print("k")

		# print(centroid_points)
	return clusters, centroid_points

def Intersection(lst1, lst2): 
		lst3 = [value for value in lst1 if value in lst2] 
		return lst3 

def get_jaccard_index(data,classes,clusters):

	dimensions = len(data[0]) - 1
	# print(dimensions)

	ground_truth_clusters = [[] for x in range(3)]
	for idx in range(len(classes)):
		for idy in range(len(data)):
			if data[idy][dimensions] == classes[idx]:
				ground_truth_clusters[idx].append(data[idy][0:len(data[idy])-1])

	# print(ground_truth_clusters)
	# print(clusters)

	jaccard_distance = [[0 for i in range(len(classes))] for j in range(len(classes))]

	for idx in range(len(classes)):
		for idy in range(len(classes)):
			list1 = clusters[idx]
			list2 = ground_truth_clusters[idy]
			intersection_list = len(Intersection(list1,list2))
			union_list = len(list1) + len(list2) - intersection_list
			jaccard_distance[idx][idy] = (union_list - intersection_list) / (union_list + 1e-7)

	# print(jaccard_distance)

	return jaccard_distance

if __name__ == '__main__':

	############### read data 
	with open('data4_19.csv', 'r') as f:
		reader = csv.reader(f)
		data = list(reader)
	# print(data)

	###############

	################ cleaning the data
	temp_data = []
	for row in data:
		if len(row) != 0:
			temp_data.append(row)
	data = []
	for row in temp_data:
		data.append([float(row[0]),float(row[1]),float(row[2]),float(row[3]),row[4]])

	################# 

	################# k - means training
	classes = []
	for row in data:
		if len(row) == 5:
			classes.append(row[4])
	classes = unique(classes)
	print('Classes: ',classes)
	print(' ')
	clusters, centroid_points = k_means(data,classes,10)

	##################
	print("___________________________________________________________")
	print("                      Cluster Means                        ")
	print("___________________________________________________________\n")
	for idx in range(len(centroid_points)):
		print('Centroid '+str(idx+1)+':', end =" ")
		for idy in range(len(centroid_points[idx])):
			print("{:.4f}".format(centroid_points[idx][idy]), end =" ")
		print(' ')
	print("___________________________________________________________")
	# print(centroid_points)

	# find Jaccard distances
	jaccard_distance = get_jaccard_index(data, classes, clusters)
	# print(' ')
	print("___________________________________________________________")
	print("                    Jaccard Distance                        ")
	print("___________________________________________________________\n")
	print("                        Ground Truth Clusters")
	print("              Iris-setosa   Iris-versicolor  Iris-virginica")
	for idx in range(3):
		print('Cluster '+str(idx+1)+':', end =" ")
		for idy in range(3):
			print("        {:.4f}".format(jaccard_distance[idx][idy]), end =" ")
		print(' ')  
	print("___________________________________________________________")
	# print(jaccard_distance)

