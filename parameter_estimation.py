import numpy as np

def gaussian_params(var_matrix):
	'''
	var_matrix is a numpy matrix of shape (n_random_variables, n_samples)
	returns a numpy covariance matrix of shape (n_random_variables, n_random_variables) and
	an n_random_variables length mean array
	'''
	shape = var_matrix.shape
	n_random_variables, n_samples = shape
	mean_array = np.apply_along_axis(np.mean, axis = 1, arr = var_matrix)
	mean_column = np.matrix(mean_array).transpose()
	# covariance_matrix = np.zeros((n_random_variables, n_random_variables))
	# for i, row in enumerate(var_matrix.transpose()):
	# 	print i
	# 	column = row.transpose()
	# 	difference = column - mean_column
	# 	product = difference*difference.transpose()
	# 	covariance_matrix += product
	one = np.ones((1, n_samples))
	demeaned_matrix = var_matrix - mean_column*one
	covariance_matrix = demeaned_matrix*demeaned_matrix.T
	covariance_matrix /= n_samples - 1
	return mean_array, covariance_matrix

# var_matrix = np.matrix([[2,4,5,3],[7,8,9,1],[2,1,6,5]])
# # var_matrix = np.matrix([[1,2,4,5]])
# mean_array, covariance_matrix = gaussian_params(var_matrix)
# print mean_array
# print covariance_matrix