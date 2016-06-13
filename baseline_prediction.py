from matrix import extrapolate
from load import movies
import numpy as np

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
	valence_movie_matrices = []
	arousal_movie_matrices = []
	diff_valence_labels_movies = []
	diff_arousal_labels_movies = []
	for i, movie in enumerate(movies):
		valence_matrix 		= np.load('../movie_matrices/%s_valence_%d.npy' % (movie, valence_t))
		arousal_matrix 		= np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, arousal_t))
		valence_labels 		= valence_labels_movies[i]
		valence_predictions = valence_model.predict(valence_matrix)
		arousal_labels 		= arousal_labels_movies[i]
		arousal_predictions = arousal_model.predict(arousal_matrix)
		valence_predictions = extrapolate(len(valence_labels), valence_predictions)
		arousal_predictions = extrapolate(len(arousal_labels), arousal_predictions)
		diff_valence_labels = valence_labels - valence_predictions
		diff_arousal_labels = arousal_labels - arousal_predictions
		diff_valence_labels_movies.append(diff_valence_labels)
		diff_arousal_labels_movies.append(diff_arousal_labels)
	return diff_valence_labels_movies, diff_arousal_labels_movies