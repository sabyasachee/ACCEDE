import load
import matrix
import numpy
from multiprocessing import Process

def save_matrices(t_array):
	'''
		load the valence and arousal labels
		load the valence and arousal correlations
		load movie_matrices each of the form (feature, frames). It is a list of lists
		for each movie
			get the window matrix: window_matrix is of the form list of tuples. each tuple of the form-
				(feature_name, statistic_number, vector) vector is caluclated on a window of t for each in
				t_array
			sort the tuples based on correlation values of (feature_name, statistic_number) and filter 
				it to have only values. number of rows increase 9 times
			get the transpose and save it in the form (seconds (samples), features)
	'''
	valence_labels, arousal_labels = load.load_output_movies()
	movies = load.movies
	movies_matrix = []
	for movie in movies:
		movies_matrix.append(load.load_input_movie(movie))
	_, _, valence_correlations, arousal_correlations = load.load_output()

	for i, movie_matrix in enumerate(movies_matrix):
		print movies[i]
		n_annotations = min(len(arousal_labels[i]), len(valence_labels[i]))
		for t in t_array:
			print t
			transformed_movie_matrix = matrix.window_matrix(movie_matrix, t, n_annotations)

			arousal_movie_matrix = matrix.sort_matrix(transformed_movie_matrix, arousal_correlations)
			valence_movie_matrix = matrix.sort_matrix(transformed_movie_matrix, valence_correlations)
			
			arousal_movie_matrix = numpy.array(arousal_movie_matrix, dtype = 'float').transpose()
			valence_movie_matrix = numpy.array(valence_movie_matrix, dtype = 'float').transpose()

			numpy.save("../movie_matrices/" + movies[i] + "_valence_" + str(t) + ".npy", valence_movie_matrix)
			numpy.save("../movie_matrices/" + movies[i] + "_arousal_" + str(t) + ".npy", arousal_movie_matrix)

# save_matrices([8, 9, 10, 11, 12, 15, 20, 30, 50, 100])
# save_matrices([1])
# t_array = [2,3,4,5,6,7]
t_array = [3,4,5]
processes = []
for t in t_array:
	process = Process(target = save_matrices, args = ([t],))
	processes.append(process)
	process.start()

for i, process in enumerate(processes):
	process.join()
	print 'Process %d joined' % i