import load
import matrix
import numpy

def save_matrices(t_array):
	
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
# save_matrices([])
save_matrices([1])