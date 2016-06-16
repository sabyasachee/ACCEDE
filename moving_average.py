import numpy as np
from matrix import extrapolate

def moving_average(matrix, window_length):
	'''
		matrix = (samples, features)
		moving average on columns of matrix
	'''
	matrix = np.atleast_2d()
	n_samples, n_features = matrix.shape
	for column in matrix.T:
		cumsum = np.cumsum(np.insert(column, 0, 0))
		averaged_column = (cumsum[window_length:] - cumsum[:-window_length])/window_length
		averaged_column = 