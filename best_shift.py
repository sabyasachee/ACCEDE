import numpy as np
from mutual_information import mutual_information

def best_shift(labels, features, shifts):
	'''
	labels is an array of length n_samples
	features is a n_features, n_samples matrix 
	shift labels backward by delta in shifts and compute mutual information
	maximum mutual information is best shift 
	'''
	labels = np.atleast_1d(labels)
	features = np.atleast_2d(features)
	best_shift = None
	max_I = None
	for shift in shifts:
		shifted_labels = labels[shift:]
		if shift:
			shifted_features = (features.T[:-shift]).T
		else:
			shifted_features = features.copy()
		I = mutual_information(shifted_labels, shifted_features)
		if max_I is None or max_I < I:
			max_I = I
			best_shift = shift
	return best_shift

# l = np.array([1,3,1,5,6,7,8,9,10,2])
# f = np.matrix([[2,3,4,1,8,7,2,3,1,3],
# 	[6,7,1,1,1,2,9,4,5,1],
# 	[4,5,1,2,3,7,3,0,2,1],
# 	[5,2,1,3,8,8,0,2,1,3]])
# print best_shift(l, f, [0,1,2])

