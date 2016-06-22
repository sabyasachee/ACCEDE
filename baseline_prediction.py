import numpy as np
import math
from matrix import extrapolate, join_vectors
from load import movies, load_output_movies
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

def baseline_prediction(valence_t, arousal_t, valence_model, arousal_model, valence_labels_movies, 
	arousal_labels_movies):
	'''
		give the predictions of valence and arousal for movies with model trained on videos
		for each movie load the corresponding matrix based on t
		predict using model trained on data1
		extrapolate to have the same shape on both labels and predictions
		return these predictions
	'''
	for i in range(len(movies)):
		n_annotations = min(len(valence_labels_movies[i]), len(arousal_labels_movies[i]))
		valence_labels_movies[i] = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')
		arousal_labels_movies[i] = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')

	valence_movie_matrices = []
	arousal_movie_matrices = []
	# diff_valence_labels_movies = []
	# diff_arousal_labels_movies = []
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
		# diff_valence_labels = valence_labels - valence_predictions
		# diff_arousal_labels = arousal_labels - arousal_predictions
		# diff_valence_labels_movies.append(diff_valence_labels)
		# diff_arousal_labels_movies.append(diff_arousal_labels)
	# all_valence_predictions = join_vectors(all_valence_predictions)
	# all_arousal_predictions = join_vectors(all_arousal_predictions)
	# all_valence_labels = join_vectors(valence_labels_movies)
	# all_arousal_labels = join_vectors(arousal_labels_movies)
	# print 'Predicting movies with model trained on video clips'
	# print 'valence rmse = %f coeff = %f' % (math.sqrt(mean_squared_error(all_valence_labels, all_valence_predictions)), 
		# np.corrcoef(all_valence_labels, all_valence_predictions)[0][1])
	# print 'arousal rmse = %f coeff = %f' % (math.sqrt(mean_squared_error(all_arousal_labels, all_arousal_predictions)), 
		# np.corrcoef(all_arousal_labels, all_arousal_predictions)[0][1])
	# return diff_valence_labels_movies, diff_arousal_labels_movies
	return all_valence_predictions, all_arousal_predictions

if __name__ == '__main__':
	valence_model, arousal_model = joblib.load('valence_model_ElasticNetCV.pkl'), \
		joblib.load('arousal_model_ElasticNetCV.pkl')
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	valence_predictions_movies, arousal_predictions_movies = baseline_prediction(10, 10, valence_model, 
		arousal_model, valence_labels_movies, arousal_labels_movies)
	valence_labels = np.hstack(tuple(valence_labels_movies))
	arousal_labels = np.hstack(tuple(arousal_labels_movies))
	# for i, movie in enumerate(movies):
	# 	with open('../H0_predictions/%s_valence_%d.txt' % (movie, 10), 'w') as fw:
	# 		fw.write('second\tprediction\n')
	# 		for j, pred in enumerate(valence_predictions_movies[i]):
	# 			fw.write('%d\t%f\n' % (j, pred))
	# 	with open('../H0_predictions/%s_arousal_%d.txt' % (movie, 10), 'w') as fw:
	# 		fw.write('second\tprediction\n')
	# 		for j, pred in enumerate(arousal_predictions_movies[i]):
	# 			fw.write('%d\t%f\n' % (j, pred))
	# with open('../H0_predictions/H0_valence_predictions.txt', 'w') as f:
	# 	f.write('film\tsecond\tprediction\n')
	# 	for i, movie in enumerate(movies):
	# 		for j, pred in enumerate(valence_predictions_movies[i]):
	# 			f.write('%d\t%d\t%f\n' % (i, j, pred))
	# with open('../H0_predictions/H0_arousal_predictions.txt', 'w') as f:
	# 	f.write('film\tsecond\tprediction\n')
	# 	for i, movie in enumerate(movies):
	# 		for j, pred in enumerate(arousal_predictions_movies[i]):
	# 			f.write('%d\t%d\t%f\n' % (i, j, pred))
	# smooths = [5,10,20,30,50,75,100,150,200,250,300]
	# valence_smooth = None
	# arousal_smooth = None
	# smoothened_valence_predictions_movies, smoothened_arousal_predictions_movies = np.arange(len(movies)), \
	# 	np.arange(len(movies))
	# for smooth in smooths:
	# 	for i in range(len(movies)):
	# 		smooth = min(smooth, len(valence_predictions_movies[i]))
	# 		smoothened_valence_predictions_movies[i] = np.convolve(valence_predictions_movies[i], 
	# 			np.ones((smooth,))/smooth, mode = 'same')
	# 		smoothened_arousal_predictions_movies[i] = np.convolve(arousal_predictions_movies[i], 
	# 			np.ones((smooth,))/smooth, mode = 'same')
	# 	valence_predictions = np.hstack(tuple(smoothened_valence_predictions_movies))
	# 	arousal_predictions = np.hstack(tuple(smoothened_arousal_predictions_movies))
	# 	print np.corrcoef(valence_predictions, valence_labels)

