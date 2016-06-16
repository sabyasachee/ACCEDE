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
	xvar_matrix = np.atleast_2d(xvar)
	yvar_matrix = np.atleast_2d(yvar)
	xyvar_matrix = np.vstack((xvar_matrix, yvar_matrix))
	n_random_variables_x = xvar_matrix.shape[0]
	n_random_variables_y = yvar_matrix.shape[0]
	n_random_variables_xy = n_random_variables_x + n_random_variables_y
	mean_x, covariance_x = gaussian_params(xvar_matrix)
	mean_y, covariance_y = gaussian_params(yvar_matrix)
	mean_xy, covariance_xy = gaussian_params(xyvar_matrix)
	# print 'y_var min %f max %f diagonal min %f max %f' % (covariance_y.min(), covariance_y.max(), 
	# 	covariance_y.diagonal().min(), covariance_y.diagonal().max())
	# print 'y_var mean %f std %f diagonal mean %f std %f' % (covariance_y.mean(), covariance_y.std(),
	# 	covariance_y.diagonal().mean(), covariance_y.diagonal().std())
	# print 'xy_var min %f max %f diagonal min %f max %f' % (covariance_xy.min(), covariance_xy.max(), 
	# 	covariance_xy.diagonal().min(), covariance_xy.diagonal().max())
	# print 'xy_var mean %f std %f diagonal mean %f std %f' % (covariance_xy.mean(), covariance_xy.std(),
	# 	covariance_xy.diagonal().mean(), covariance_xy.diagonal().std())
	covariance_y += 0.001*np.identity(n_random_variables_y)
	covariance_xy += 0.001*np.identity(n_random_variables_xy)
	_, logdet_x = np.linalg.slogdet(covariance_x)
	_, logdet_y = np.linalg.slogdet(covariance_y)
	_, logdet_xy = np.linalg.slogdet(covariance_xy)
	constant = np.log(2*np.pi*np.e)
	# print logdet_x, logdet_y, logdet_xy
	try:
		x_entropy = 0.5*(n_random_variables_x * constant + logdet_x)
		y_entropy = 0.5*(n_random_variables_y * constant + logdet_y)
		xy_entropy = 0.5*(n_random_variables_xy * constant + logdet_xy)
		I = x_entropy + y_entropy - xy_entropy
	except Exception:
		I = x_entropy
	return I

# a = np.matrix([[2,1,4],[1,5,6],[5,6,7]])
# b = np.matrix([[12,32,41],[11,55,62],[51,62,73]])
# print mutual_information(a, b)