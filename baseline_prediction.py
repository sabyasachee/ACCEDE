import numpy as np
import math
from matrix import extrapolate, join_vectors
from load import movies
from sklearn.metrics import mean_squared_error

def baseline_prediction(valence_t, arousal_t, valence_model, arousal_model, valence_labels_movies, 
	arousal_labels_movies):
	'''
		give the predictions of valence and arousal for movies with model trained on videos
		for each movie load the corresponding matrix based on t
		predict using model trained on data1
		extrapolate to have the same shape on both labels and predictions
		calculate the difference: true - predict
		return this error difference
	'''
	for i in range(len(movies)):
		n_annotations = min(len(valence_labels_movies[i]), len(arousal_labels_movies[i]))
		valence_labels_movies[i] = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')
		arousal_labels_movies[i] = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')

	valence_movie_matrices = []
	arousal_movie_matrices = []
	diff_valence_labels_movies = []
	diff_arousal_labels_movies = []
	all_valence_predictions = []
	all_arousal_predictions = []
	for i, movie in enumerate(movies):
		valence_matrix 		= np.load('../movie_matrices/%s_valence_%d.npy' % (movie, valence_t))
		arousal_matrix 		= np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, arousal_t))
		valence_labels 		= valence_labels_movies[i]
		valence_predictions = valence_model.predict(valence_matrix)
		arousal_labels 		= arousal_labels_movies[i]
		arousal_predictions = arousal_model.predict(arousal_matrix)
		valence_predictions = (valence_predictions - 1)/2 - 1
		arousal_predictions = (arousal_predictions - 1)/2 - 1
		valence_predictions = extrapolate(len(valence_labels), valence_predictions)
		arousal_predictions = extrapolate(len(arousal_labels), arousal_predictions)
		all_valence_predictions.append(valence_predictions)
		all_arousal_predictions.append(arousal_predictions)
		diff_valence_labels = valence_labels - valence_predictions
		diff_arousal_labels = arousal_labels - arousal_predictions
		diff_valence_labels_movies.append(diff_valence_labels)
		diff_arousal_labels_movies.append(diff_arousal_labels)
	all_valence_predictions = join_vectors(all_valence_predictions)
	all_arousal_predictions = join_vectors(all_arousal_predictions)
	all_valence_labels = join_vectors(valence_labels_movies)
	all_arousal_labels = join_vectors(arousal_labels_movies)
	print 'Predicting movies with model trained on video clips'
	print 'valence rmse = %f coeff = %f' % (math.sqrt(mean_squared_error(all_valence_labels, all_valence_predictions)), 
		np.corrcoef(all_valence_labels, all_valence_predictions)[0][1])
	print 'arousal rmse = %f coeff = %f' % (math.sqrt(mean_squared_error(all_arousal_labels, all_arousal_predictions)), 
		np.corrcoef(all_arousal_labels, all_arousal_predictions)[0][1])
	return diff_valence_labels_movies, diff_arousal_labels_movies