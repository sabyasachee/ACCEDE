import math
import numpy as np
from load import movies, load_output_movies
from matrix import join_vectors, join_matrices
from multiprocessing import Process

def correlations(t_array):
	np.set_printoptions(precision = 2, linewidth = 100)
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	for t in t_array:
		print t
		valence_labels, arousal_labels, valence_matrices, arousal_matrices, valence_header, \
			arousal_header = [], [], [], [], [], []

		with open('../valence_correlations_videos.txt') as fr:
			for line in fr:
				feature, statistic, _ = line.strip().split('\t')
				statistic = int(statistic)
				valence_header.append([feature, statistic, None])
		with open('../arousal_correlations_videos.txt') as fr:
			for line in fr:
				feature, statistic, _ = line.strip().split('\t')
				statistic = int(statistic)
				arousal_header.append([feature, statistic, None])

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
		print "%d Joining vectors" % t
		valence_labels, arousal_labels = join_vectors(valence_labels), join_vectors(arousal_labels)
		print "%d Joining matrices" % t
		valence_matrix, arousal_matrix = join_matrices(valence_matrices).T, join_matrices(arousal_matrices).T
		print t, valence_labels.shape, arousal_labels.shape, valence_matrix.shape, arousal_matrix.shape
		aug_valence_matrix = np.vstack((valence_labels, valence_matrix))
		aug_arousal_matrix = np.vstack((arousal_labels, arousal_matrix))

		print "%d Finding correlation valence" % t
		valence_correlations = np.corrcoef(aug_valence_matrix)[0][1:]
		valence_correlations[np.isnan(valence_correlations)] = -10
		for i, valence_correlation in enumerate(valence_correlations):
			valence_header[i][2] = valence_correlation
		print "%d Finding correlation arousal" % t
		arousal_correlations = np.corrcoef(aug_arousal_matrix)[0][1:]
		arousal_correlations[np.isnan(arousal_correlations)] = -10
		for i, arousal_correlation in enumerate(arousal_correlations):
			arousal_header[i][2] = arousal_correlation
		valence_correlations[::-1].sort()
		arousal_correlations[::-1].sort()
		valence_header = sorted(valence_header, key = lambda valence_header_tuple: valence_header_tuple[2], 
			reverse = True)
		arousal_header = sorted(arousal_header, key = lambda arousal_header_tuple: arousal_header_tuple[2], 
			reverse = True)
		with open('../movie_results/valence_correlations_%d.txt' % t, 'w') as fw:
			for feature, statistic, correlation in valence_header:
				fw.write('%s\t%d\t%f\n' % (feature, statistic, correlation))
		with open('../movie_results/arousal_correlations_%d.txt' % t, 'w') as fw:
			for feature, statistic, correlation in arousal_header:
				fw.write('%s\t%d\t%f\n' % (feature, statistic, correlation))
		# print t, "valence", valence_correlations[:50]
		# print t, "arousal", arousal_correlations[:50]
		print t, "Done"

# processes = []
# t_array = [1,2,3,4,5,6,7,8,9,10]
# for t in t_array:
# 	process = Process(target = correlations, args = ([t],))
# 	processes.append(process)
# 	process.start()

# for i, process in enumerate(processes):
# 	process.join()
# 	print "Process %d joined" % (i + 1)

correlations([1,2,3,4,5,6,7,8,9,10])