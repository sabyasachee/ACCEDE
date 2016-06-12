import numpy as np
from parameter_estimation import gaussian_params

def mutual_information(xvar_matrix, yvar_matrix):
	'''
	xvar_matrix is a matrix of shape (n_random_variables_x, n_samples)
	yvar_matrix is a matrix of shape (n_random_variables_y, n_samples)
	calculate normal parameters of x_var, y_var and x_var | y_var
	calculate entropies
	return mutual information
	'''
	xvar_matrix = np.atleast_2d(xvar_matrix)
	yvar_matrix = np.atleast_2d(yvar_matrix)
	xyvar_matrix = np.vstack((xvar_matrix, yvar_matrix))
	n_random_variables_x = xvar_matrix.shape[0]
	n_random_variables_y = yvar_matrix.shape[0]
	n_random_variables_xy = n_random_variables_x + n_random_variables_y
	mean_x, covariance_x = gaussian_params(xvar_matrix)
	mean_y, covariance_y = gaussian_params(yvar_matrix)
	mean_xy, covariance_xy = gaussian_params(xyvar_matrix)
	print covariance_x
	print covariance_y
	print covariance_xy
	x_entropy = 0.5*np.log(np.power(2*np.pi*np.e, n_random_variables_x) * \
		np.absolute(np.linalg.det(covariance_x)))
	y_entropy = 0.5*np.log(np.power(2*np.pi*np.e, n_random_variables_y) * \
		np.absolute(np.linalg.det(covariance_y)))
	xy_entropy = 0.5*np.log(np.power(2*np.pi*np.e, n_random_variables_xy) * \
		np.absolute(np.linalg.det(covariance_xy)))
	return x_entropy + y_entropy - xy_entropy

# a = np.matrix([[2,1,4],[1,5,6],[5,6,7]])
# b = np.matrix([[12,32,41],[11,55,62],[51,62,73]])
# print mutual_information(a, b)