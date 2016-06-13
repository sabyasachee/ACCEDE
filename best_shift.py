import numpy as np
from mutual_information import mutual_information
from load import movies, load_output_movies

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
	return best_shift, max_I

def best_shift_movies(labels_movies, features_movies, shifts):
	'''
		labels_movies is an array of vectors each of shape n_samples,
		features_movies is an array of matrices each of shape (n_samples(seconds), n_features)
		shifts is an array of shifts
		get the best shift for these movies
		The best shift is the one which has the highest sum of mutual independence on all movies
	'''
	n_movies = len(labels_movies)
	I_s = np.zeros((n_movies))
	for labels_movie, features_movie in zip(labels_movies, features_movies):
		for i, shift in enumerate(shifts):
			features = features_movies.T
			shifted_labels = labels_movie[shift:]
			if shift:
				shifted_features = (features.T[:-shift]).T
			else:
				shifted_features = features.copy()
			I_s[i] += mutual_information(shifted_labels, shifted_features)
	best_shift = None
	max_I = None
	for i, I in enumerate(I_s):
		if max_I is None or max_I < I:
			best_shift = shifts[i]
	return best_shift

# l = np.array([1,3,1,5,6,7,8,9,10,2])
# f = np.matrix([[2,3,4,1,8,7,2,3,1,3],
# 	[6,7,1,1,1,2,9,4,5,1],
# 	[4,5,1,2,3,7,3,0,2,1],
# 	[5,2,1,3,8,8,0,2,1,3]])
# print best_shift(l, f, [0,1,2])
if __name__ == '__main__':
	valence_labels_movies, arousal_labels_movies = load_output_movies()


