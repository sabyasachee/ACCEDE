from utils import get_statistics
from load import feature_names
import numpy as np
import math


def window_matrix(matrix, window_length, n_annotations):
	new_matrix = []
	for i in range(0, len(matrix)):
		feature_name = feature_names[i]
		n_frames = len(matrix[i])

		mean_row = []
		median_row = []
		std_row = []
		kurtosis_row = []
		lower_quartile_row = []
		upper_quartile_row = []
		min_row = []
		max_row = []
		range_row = []

		annotations_borders = []
		for j in range(0, n_annotations):
			start_frame = (j * n_frames)/n_annotations
			end_frame = ((j + 1) * n_frames)/n_annotations
			annotations_borders.append((start_frame, end_frame))

		for j in range(0, n_annotations - window_length + 1):
			start_frame = annotations_borders[j][0]
			end_frame = annotations_borders[j + window_length - 1][1]
			vector = matrix[i][start_frame: end_frame]
			mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range = \
				get_statistics(vector, clip = False)
			mean_row.append(mean)
			median_row.append(median)
			std_row.append(std)
			kurtosis_row.append(kurtosis)
			lower_quartile_row.append(lower_quartile)
			upper_quartile_row.append(upper_quartile)
			min_row.append(_min)
			max_row.append(_max)
			range_row.append(_range)
		new_matrix.append((feature_name, 0, mean_row))
		new_matrix.append((feature_name, 1, median_row))
		new_matrix.append((feature_name, 2, std_row))
		new_matrix.append((feature_name, 3, kurtosis_row))
		new_matrix.append((feature_name, 4, lower_quartile_row))
		new_matrix.append((feature_name, 5, upper_quartile_row))
		new_matrix.append((feature_name, 6, min_row))
		new_matrix.append((feature_name, 7, max_row))
		new_matrix.append((feature_name, 8, range_row))
	return new_matrix

def sort_matrix(matrix, correlations):
	matrix = sorted(matrix, key = lambda row_tuple: correlations[row_tuple[0]][row_tuple[1]], 
		reverse = True)
	new_matrix = []
	for name, statistic, row in matrix:
		new_matrix.append(row)
	# new_matrix = clip_matrix(new_matrix)
	return new_matrix
		
def clip_matrix(matrix):
	min_length = 999999999
	new_matrix = []
	for row in matrix:
		length = len(row)
		if length < min_length:
			min_length = length
	for row in matrix:
		new_row = matrix[:min_length]
		new_matrix.append(new_row)
	return new_matrix

def join_matrices(matrices):
	result_matrix = None
	for matrix in matrices:
		if result_matrix is None:
			result_matrix = matrix.copy()
		else:
			result_matrix = np.concatenate((result_matrix, matrix))
	return result_matrix

def join_vectors(vectors):
	result_vector = np.array([], dtype = 'float')
	for vector in vectors:
		result_vector = np.append(result_vector, vector)
	return result_vector

def rmse_matrix(vector1, vector2):
	difference = vector2 - vector1
	product = difference*difference
	mean = product.mean()
	rmse = math.sqrt(mean)
	return rmse

# a = np.array([1,2,3])
# b = np.array([3,4,5])
# c = np.array([1,2])
# print join_vectors([a,b,c])