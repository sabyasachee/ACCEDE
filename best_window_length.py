import numpy as np
import math
from load import movies, load_output_movies
from matrix import join_vectors, join_matrices
from mutual_information import mutual_information
from multiprocessing import Process

def best_window_length(t_array):
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	valence_t, arousal_t, valence_I, arousal_I = None, None, None, None
	for t in t_array:
		print t
		valence_labels, arousal_labels, valence_matrices, arousal_matrices = [], [], [], []
		for i, movie in enumerate(movies):
			if i and not i %5:
				print "%d Processing movie %d" % (t, i)
			n_annotations = min(len(valence_labels_movies[i]), len(arousal_labels_movies[i]))
			valence_labels_movie = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')
			arousal_labels_movie = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')
			if t > 2:
				if t % 2:
					arousal_labels_movie = arousal_labels_movie[t/2:-(t/2)]
					valence_labels_movie = valence_labels_movie[t/2:-(t/2)]
				else:
					arousal_labels_movie = arousal_labels_movie[t/2:-(t/2) + 1]
					valence_labels_movie = valence_labels_movie[t/2:-(t/2) + 1]
			elif t == 2:
				valence_labels_movie = valence_labels_movie[1:]
				arousal_labels_movie = arousal_labels_movie[1:]
			valence_labels.append(valence_labels_movie)
			arousal_labels.append(arousal_labels_movie)
			valence_matrix = np.load('../movie_matrices/%s_valence_%d.npy' % (movie, t))
			arousal_matrix = np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, t))
			valence_matrices.append(valence_matrix)
			arousal_matrices.append(arousal_matrix)
		valence_labels, arousal_labels = join_vectors(valence_labels), join_vectors(arousal_labels)
		valence_matrices, arousal_matrices = join_matrices(valence_matrices), join_matrices(arousal_matrices)
		print t, valence_labels.shape, valence_matrices.shape
		print t, arousal_labels.shape, arousal_matrices.shape
		v_I = mutual_information(valence_labels, valence_matrices.T, t)
		a_I = mutual_information(arousal_labels, arousal_matrices.T, t)
		print v_I, a_I
		if valence_I is None or valence_I < v_I:
			valence_I = v_I
			valence_t = t
		if arousal_I is None or arousal_I < a_I:
			arousal_I = a_I
			arousal_t = t
	print "Done"
	print "valence_t %d, arousal_t %d, valence_I %f, arousal_I %f" % (valence_t, arousal_t, valence_I, 
		arousal_I)


# print best_window_length([1,2,4,5,6,7,8,9,10])
processes = []
t_array = [1,2,3,4,5,6,7,8,9,10]
for t in t_array:
	process = Process(target = best_window_length, args = ([t],))
	processes.append(process)
	process.start()

for i, process in enumerate(processes):
	process.join()
	print "Process %d joined" % i + 1
