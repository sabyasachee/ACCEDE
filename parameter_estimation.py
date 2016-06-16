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
	one = np.ones((1, n_samples))
	demeaned_matrix = var_matrix - mean_column*one
	covariance_matrix = demeaned_matrix*demeaned_matrix.T
	covariance_matrix /= n_samples - 1
	return mean_array, covariance_matrix

