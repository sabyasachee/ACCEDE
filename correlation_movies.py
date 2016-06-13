import math
import numpy as np
from load import movies, load_output_movies
from matrix import join_vectors, join_matrices

def correlations(t_array):
	n_features = 1404
	for t in t_array:
		valence_labels_movies, arousal_labels_movies = load_output_movies()
		valence_matrices = []
		arousal_matrices = []
		for i, movie in enumerate(movies):
			n_labels = min(len(valence_labels_movies[i]), len(arousal_labels_movies[i]))
			valence_labels_movies[i] = valence_labels_movies[i][:n_labels]
			arousal_labels_movies[i] = arousal_labels_movies[i][:n_labels]
			if t > 1:
				if t % 2:
					arousal_labels_movies[i] = np.array(arousal_labels_movies[i][t/2:-(t/2)], dtype = 'float')
					valence_labels_movies[i] = np.array(valence_labels_movies[i][t/2:-(t/2)], dtype = 'float')
				else:
					arousal_labels_movies[i] = np.array(arousal_labels_movies[i][t/2:-(t/2) + 1], dtype = 'float')
					valence_labels_movies[i] = np.array(valence_labels_movies[i][t/2:-(t/2) + 1], dtype = 'float')
			valence_matrix = np.load('../movie_matrices/%s_valence_%d.npy' % (movie, t))
			arousal_matrix = np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, t))
			valence_matrices.append(valence_matrix)
			arousal_matrices.append(arousal_matrix)
		valence_labels, arousal_labels = join_vectors(valence_labels_movies), join_vectors(arousal_labels_movies)
		valence_matrix = join_matrices(valence_matrices).T
		arousal_matrix = join_matrices(arousal_matrices).T
		valence_matrix = np.vstack((valence_labels, valence_matrix))
		arousal_matrix = np.vstack((arousal_labels, arousal_matrix))
		print t
		valence_correlations = np.diag(np.fliplr(np.corrcoef(valence_matrix)))
		arousal_correlations = np.diag(np.fliplr(np.corrcoef(arousal_matrix)))
		valence_correlations.sort()
		arousal_correlations.sort()
		print valence_correlations[-20:]
		print arousal_correlations[-20:]

correlations([1,8,9,10,12,15])