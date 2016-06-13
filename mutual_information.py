import numpy as np
from parameter_estimation import gaussian_params

def mutual_information(xvar, yvar):
	'''
	xvar_matrix is a matrix of shape (n_random_variables_x, n_samples)
	yvar_matrix is a matrix of shape (n_random_variables_y, n_samples)
	calculate normal parameters of x_var, y_var and x_var | y_var
	calculate entropies
	return mutual information
	'''
	# np.set_printoptions(edgeitems = 30, linewidth = 150)
	xvar_matrix = np.atleast_2d(xvar)
	yvar_matrix = np.atleast_2d(yvar)
	xyvar_matrix = np.vstack((xvar_matrix, yvar_matrix))
	n_random_variables_x = xvar_matrix.shape[0]
	n_random_variables_y = yvar_matrix.shape[0]
	n_random_variables_xy = n_random_variables_x + n_random_variables_y
	print "Estimating parameters for xvar"
	mean_x, covariance_x = gaussian_params(xvar_matrix)
	print "Estimating parameters for yvar"
	mean_y, covariance_y = gaussian_params(yvar_matrix)
	# print np.linalg.slogdet(covariance_y)
	print "Estimating parameters for xvar, yvar"
	mean_xy, covariance_xy = gaussian_params(xyvar_matrix)
	# print np.linalg.slogdet(covariance_xy)
	print "Parameter Estimation done"
	_, logdet_x = np.linalg.slogdet(covariance_x)
	_, logdet_y = np.linalg.slogdet(covariance_y)
	_, logdet_xy = np.linalg.slogdet(covariance_xy)
	print "Log Determinants x = %f y = %f xy = %f" % (logdet_x, logdet_y, logdet_xy)
	constant = np.log(2*np.pi*np.e)
	x_entropy = 0.5*(n_random_variables_x * constant + logdet_x)
	y_entropy = 0.5*(n_random_variables_y * constant + logdet_y)
	xy_entropy = 0.5*(n_random_variables_xy * constant + logdet_xy)
	return x_entropy + y_entropy - xy_entropy

# a = np.matrix([[2,1,4],[1,5,6],[5,6,7]])
# b = np.matrix([[12,32,41],[11,55,62],[51,62,73]])
# print mutual_information(a, b)