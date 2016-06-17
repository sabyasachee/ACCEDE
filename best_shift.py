import numpy as np
from mutual_information import mutual_information
from load import movies, load_output_movies

def shift_values(labels, features, shifts):
	'''
	labels is an array of length n_samples
	features is a n_features, n_samples matrix 
	shift labels backward by delta in shifts and compute mutual information
	calculate mutual information for each shift
	return values
	'''
	labels = np.atleast_1d(labels)
	features = np.atleast_2d(features)
	I_s = np.empty(len(shifts))
	for i, shift in enumerate(shifts):
		shifted_labels = labels[shift:]
		if shift:
			shifted_features = features[:,:-shift]
		else:
			shifted_features = features.copy()
		I = mutual_information(shifted_labels, shifted_features)
		I_s[i] = I
	return I_s

def best_shift(valence_matrices, arousal_matrices, all_valence_labels, all_arousal_labels, valence_shifts, 
	arousal_shifts):
	valence_sum_Is, arousal_sum_Is = np.zeros(len(valence_shifts)), np.zeros(len(arousal_shifts))
	for i in range(len(movies)):
		valence_matrix, arousal_matrix = valence_matrices[i], arousal_matrices[i]
		valence_labels, arousal_labels = all_valence_labels[i], all_arousal_labels[i]
		valence_Is = shift_values(valence_labels, valence_matrix.T, valence_shifts)
		arousal_Is = shift_values(arousal_labels, arousal_matrix.T, arousal_shifts)
		print valence_Is, arousal_Is
		valence_sum_Is += valence_Is
		arousal_sum_Is += arousal_Is
		print 'mutual information calculated for shifts for %s' % movies[i]
	print valence_sum_Is, arousal_sum_Is
	valence_shift = valence_shifts[np.argmax(valence_sum_Is)]
	print 'best valence shift', valence_shift
	arousal_shift = arousal_shifts[np.argmax(arousal_sum_Is)]
	print 'best arousal shift', arousal_shift
	return valence_shift, arousal_shift
	# if valence_shift:
	# 	for i in range(len(movies)):
	# 		valence_matrices[i] = valence_matrices[i][:-valence_shift,:]
	# 		all_valence_labels[i] = all_valence_labels[i][valence_shift:]
	# if arousal_shift:
	# 	for i in range(len(movies)):
	# 		arousal_matrices[i] = arousal_matrices[i][:-arousal_shift,:]
	# 		all_arousal_labels[i] = all_arousal_labels[i][arousal_shift:]
	# print 'Shift Applied'
	# return valence_matrices, arousal_matrices, all_valence_labels, all_arousal_labels