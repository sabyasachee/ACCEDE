from matrix import join_vectors, join_matrices
from load import movies
import numpy as np

def threshold_n_features(threshold, valence_t, arousal_t):
	n_valence_features, n_arousal_features = 0, 0
	with open('../movie_results/valence_correlations_%d.txt' % valence_t) as fr:
		for line in fr:
			_, _, value = line.strip().split('\t')
			value = float(value)
			if value < threshold:
				break
			else:
				n_valence_features += 1
	with open('../movie_results/arousal_correlations_%d.txt' % arousal_t) as fr:
		for line in fr:
			_, _, value = line.strip().split('\t')
			value = float(value)
			if value < threshold:
				break
			else:
				n_arousal_features += 1
	return n_valence_features, n_arousal_features

# print threshold_n_features(0.05, 1, 1)

def feature_selection(valence_matrices, arousal_matrices, valence_labels, arousal_labels, threshold):
	valence_start_rows, valence_end_rows = [], []
	arousal_start_rows, arousal_end_rows = [], []
	valence_rsum, arousal_rsum = 0, 0
	for i in range(len(movies)):
		if valence_matrices[i].shape[0] != len(valence_labels[i]) or arousal_matrices[i].shape[0] != len(arousal_labels[i]): 
			print "n_samples of labels and movies don't match for movie %d" % i
			return None, None			
		valence_start_rows.append(valence_rsum)
		valence_rsum += len(valence_matrices[i])
		valence_end_rows.append(valence_rsum)
		arousal_start_rows.append(arousal_rsum)
		arousal_rsum += len(arousal_matrices[i])
		arousal_end_rows.append(arousal_rsum)
	valence_matrix = np.vstack(tuple(valence_matrices))
	arousal_matrix = np.vstack(tuple(arousal_matrices))
	valence_labels = np.hstack(tuple(valence_labels))
	arousal_labels = np.hstack(tuple(arousal_labels))
	print 'concatenation done'
	valence_correlations = np.zeros((valence_matrix.shape[1],))
	arousal_correlations = np.zeros((arousal_matrix.shape[1],))
	for i in range(valence_matrix.shape[1]):
		valence_corr = np.corrcoef(valence_labels, valence_matrix[:,i])[0][1]
		valence_correlations[i] = valence_corr
	for i in range(arousal_matrix.shape[1]):
		arousal_corr = np.corrcoef(arousal_labels, arousal_matrix[:,i])[0][1]
		arousal_correlations[i] = arousal_corr
	n_valence_features = (valence_correlations > threshold).sum()
	n_arousal_features = (arousal_correlations > threshold).sum()
	valence_col_order = np.argsort(valence_correlations)[::-1]
	arousal_col_order = np.argsort(arousal_correlations)[::-1]
	valence_matrix = valence_matrix[:,valence_col_order]
	arousal_matrix = arousal_matrix[:,arousal_col_order]
	valence_matrix = valence_matrix[:,n_valence_features:]
	arousal_matrix = arousal_matrix[:,n_arousal_features:]
	valence_matrices, arousal_matrices = [], []
	for i in range(len(movies)):
		valence_matrices.append(valence_matrix[valence_start_rows[i]:valence_end_rows[i],:])
		arousal_matrices.append(arousal_matrix[arousal_start_rows[i]:arousal_end_rows[i],:])
	return valence_matrices, arousal_matrices