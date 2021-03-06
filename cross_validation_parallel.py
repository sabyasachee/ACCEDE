import numpy as np
import math
import sys
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from load import movies, load_output_movies
from matrix import join_matrices, join_vectors, rmse_matrix, extrapolate
from multiprocessing import Process, Array
from baseline_prediction import baseline_prediction
from sklearn.metrics import mean_squared_error
from threshold import threshold_n_features

def fold_training(valence_predictions, arousal_predictions, 
	start_fold, end_fold,
	valence_regressors, arousal_regressors, 
	valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies, 
	n_test_matrices, n_train_matrices, n_valid_matrices, 
	use_dev):

	if not use_dev:
		n_train_matrices = n_train_matrices + n_valid_matrices

	for i in range(end_fold - 1, start_fold - 1, -1):
		print "Fold %d" % i
		valence_test_matrices	= valence_movie_matrices[i*n_test_matrices: (i+1)*n_test_matrices]
		valence_test_labels		= valence_labels_movies[i*n_test_matrices: (i+1)*n_test_matrices]
		if use_dev:
			rest_valence_matrices	= valence_movie_matrices[:i*n_test_matrices] + \
				valence_movie_matrices[(i+1)*n_test_matrices:]
			rest_valence_labels		= valence_labels_movies[:i*n_test_matrices] + \
				valence_labels_movies[(i+1)*n_test_matrices:]
			valence_train_matrices	= rest_valence_matrices[:n_train_matrices]
			valence_train_labels	= rest_valence_labels[:n_train_matrices]
			valence_valid_matrices	= rest_valence_matrices[n_train_matrices:]
			valence_valid_labels	= rest_valence_labels[n_train_matrices:]
		else:
			valence_train_matrices = valence_movie_matrices[:i*n_test_matrices] + \
				valence_movie_matrices[(i+1)*n_test_matrices:]
			valence_train_labels = valence_labels_movies[:i*n_test_matrices] + \
				valence_labels_movies[(i+1)*n_test_matrices:]

		arousal_test_matrices	= arousal_movie_matrices[i*n_test_matrices: (i+1)*n_test_matrices]
		arousal_test_labels		= arousal_labels_movies[i*n_test_matrices: (i+1)*n_test_matrices]
		if use_dev:
			rest_arousal_matrices	= arousal_movie_matrices[:i*n_test_matrices] + \
				arousal_movie_matrices[(i+1)*n_test_matrices:]
			rest_arousal_labels		= arousal_labels_movies[:i*n_test_matrices] + \
				arousal_labels_movies[(i+1)*n_test_matrices:]
			arousal_train_matrices	= rest_arousal_matrices[:n_train_matrices]
			arousal_train_labels	= rest_arousal_labels[:n_train_matrices]
			arousal_valid_matrices	= rest_arousal_matrices[n_train_matrices:]
			arousal_valid_labels	= rest_arousal_labels[n_train_matrices:]
		else:
			arousal_train_matrices = arousal_movie_matrices[:i*n_test_matrices] + \
				arousal_movie_matrices[(i+1)*n_test_matrices:]
			arousal_train_labels = arousal_labels_movies[:i*n_test_matrices] + \
				arousal_labels_movies[(i+1)*n_test_matrices:]

		valence_test_matrix		= np.vstack(tuple(valence_test_matrices))  
		valence_test_labels		= np.hstack(tuple(valence_test_labels))
		valence_train_matrix	= np.vstack(tuple(valence_train_matrices))
		valence_train_labels	= np.hstack(tuple(valence_train_labels))
		if use_dev:
			valence_valid_matrix	= np.vstack(tuple(valence_valid_matrices))
			valence_valid_labels	= np.hstack(tuple(valence_valid_labels))

		arousal_test_matrix		= np.vstack(tuple(arousal_test_matrices))  
		arousal_test_labels		= np.hstack(tuple(arousal_test_labels))
		arousal_train_matrix	= np.vstack(tuple(arousal_train_matrices))
		arousal_train_labels	= np.hstack(tuple(arousal_train_labels))
		if use_dev:
			arousal_valid_matrix	= np.vstack(tuple(arousal_valid_matrices))
			arousal_valid_labels	= np.hstack(tuple(arousal_valid_labels))

		valence_offset, arousal_offset = 0, 0
		labels_movies = valence_labels_movies[:i*n_test_matrices]
		for labels_movie in labels_movies:
			valence_offset += len(labels_movie)	
		# print "Fold %d valence filling %d to %d" % (i, valence_offset, valence_offset + len(valence_test_labels) - 1)
		labels_movies = arousal_labels_movies[:i*n_test_matrices]
		for labels_movie in labels_movies:
			arousal_offset += len(labels_movie)	
		# print "Fold %d arousal filling %d to %d" % (i, arousal_offset, arousal_offset + len(arousal_test_labels) - 1)		

		valence_rmse = None
		arousal_rmse = None
		valence_model = valence_regressors[0]
		arousal_model = arousal_regressors[0]
		for regressor in valence_regressors:
			regressor.fit(valence_train_matrix, valence_train_labels)
			if use_dev:
				valence_valid_predictions = regressor.predict(valence_valid_matrix)
				rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
				if valence_rmse is None or valence_rmse > rmse:
					valence_rmse = rmse
					valence_model = regressor

		for regressor in arousal_regressors:
			regressor.fit(arousal_train_matrix, arousal_train_labels)
			if use_dev:
				arousal_valid_predictions = regressor.predict(arousal_valid_matrix)
				rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
				if arousal_rmse is None or arousal_rmse > rmse:
					arousal_rmse = rmse
					arousal_model = regressor

		if type(valence_model).__name__ == 'Ridge':
			print 'Fold %d best valence model Ridge alpha %f' % (i, valence_model.alpha)
		else:
			print 'Fold %d best valence model %s' % (i, type(valence_model).__name__)
		if type(arousal_model).__name__ == 'Ridge':
			print 'Fold %d best arousal model Ridge alpha %f' % (i, arousal_model.alpha)
		else:
			print 'Fold %d best arousal model %s' % (i, type(arousal_model).__name__)

		valence_test_predictions = valence_model.predict(valence_test_matrix)
		arousal_test_predictions = arousal_model.predict(arousal_test_matrix)
		print "Fold %d Valence rmse = %f coeff %f" % (i, math.sqrt(mean_squared_error(valence_test_predictions, 
			valence_test_labels)), np.corrcoef(valence_test_predictions, valence_test_labels)[0][1])
		print "Fold %d Arousal rmse = %f coeff %f" % (i, math.sqrt(mean_squared_error(arousal_test_predictions, 
			arousal_test_labels)), np.corrcoef(arousal_test_predictions, arousal_test_labels)[0][1])

		for j, valence_test_prediction in enumerate(valence_test_predictions):
			valence_predictions[valence_offset + j] = valence_test_prediction
		for j, arousal_test_prediction in enumerate(arousal_test_predictions):
			arousal_predictions[arousal_offset + j] = arousal_test_prediction

		print "Fold %d Done" % i

def simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies, n_folds = 10, use_dev = True):
	n_movies = len(movies)
	n_train_matrices = (n_movies/n_folds)*(((n_folds - 1)*8)/10)
	n_valid_matrices = (n_movies/n_folds)*(((n_folds - 1)*2)/10)
	n_test_matrices = n_movies/n_folds
	while n_train_matrices + n_valid_matrices + n_test_matrices < n_movies:
		n_train_matrices += 1
	print n_train_matrices, n_valid_matrices, n_test_matrices
	valence_labels = np.hstack(tuple(valence_labels_movies))
	arousal_labels = np.hstack(tuple(arousal_labels_movies))
	valence_predictions = Array('d', len(valence_labels))
	arousal_predictions = Array('d', len(arousal_labels))
	processes = []
	for i in range(0, 10):
		process = Process(target = fold_training, args = (valence_predictions, arousal_predictions, 
			(n_folds*i)/10, (n_folds*(i + 1))/10, 
			valence_regressors, arousal_regressors, 
			valence_movie_matrices, arousal_movie_matrices, 
			valence_labels_movies, arousal_labels_movies, 
			n_test_matrices, n_train_matrices, n_valid_matrices, 
			use_dev, ))
		processes.append(process)
		process.start()

	for process in processes:
		process.join()
	print "All processes joined"

	valence_predictions = np.array(valence_predictions, dtype = 'float')
	arousal_predictions = np.array(arousal_predictions, dtype = 'float')
	valence_starts, valence_ends, arousal_starts, arousal_ends = [], [], [], []
	valence_length, arousal_length = 0, 0
	for i in range(len(movies)):
		valence_starts.append(valence_length)
		valence_length += len(valence_labels_movies[i])
		valence_ends.append(valence_length)
		arousal_starts.append(arousal_length)
		arousal_length += len(arousal_labels_movies[i])
		arousal_ends.append(arousal_length)
	valence_predictions_movies, arousal_predictions_movies = [], []
	for i in range(len(movies)):
		valence_predictions_movies.append(valence_predictions[valence_starts[i]:valence_ends[i]])
		arousal_predictions_movies.append(arousal_predictions[arousal_starts[i]:arousal_ends[i]])
	return valence_predictions_movies, arousal_predictions_movies

if __name__ == '__main__':
	valence_regressors = [
						LinearRegression(), 
						# ElasticNetCV(max_iter = 10000), 
						# BayesianRidge(), 
						# ElasticNet(max_iter = 10000, alpha = 0.1), 
						# ElasticNet(max_iter = 10000, alpha = 10), 
						# DecisionTreeRegressor(max_depth = 2)
						]
	arousal_regressors = [
						LinearRegression(), 
						# ElasticNetCV(max_iter = 10000), 
						# BayesianRidge(), 
						# ElasticNet(max_iter = 10000, alpha = 0.1), 
						# ElasticNet(max_iter = 10000, alpha = 10), 
						# DecisionTreeRegressor(max_depth = 2)
						]
	valence_elasticnet_regressors = []
	arousal_elasticnet_regressors = []
	for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
		for l1_ratio in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
			elasticnet_regressor = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = 10000)
			valence_elasticnet_regressors.append(elasticnet_regressor)
			elasticnet_regressor = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, max_iter = 10000)
			arousal_elasticnet_regressors.append(elasticnet_regressor)
	valence_movie_t = int(sys.argv[1])
	arousal_movie_t = int(sys.argv[2])
	threshold = float(sys.argv[3])
	print valence_movie_t, arousal_movie_t, threshold
	arousal_movie_matrices = []
	valence_movie_matrices = []
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	for i, movie in enumerate(movies):
		n_annotations = min(len(arousal_labels_movies[i]), len(valence_labels_movies[i]))
		valence_movie_matrices.append(np.load("../movie_matrices_movie_features/%s_valence_%d.npy" % (movie, valence_movie_t)))
		arousal_movie_matrices.append(np.load("../movie_matrices_movie_features/%s_arousal_%d.npy" % (movie, arousal_movie_t)))
		valence_labels_movies[i] = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')
		arousal_labels_movies[i] = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')
		
		if valence_movie_t > 2:
			if valence_movie_t % 2:
				valence_labels_movies[i] = valence_labels_movies[i][valence_movie_t/2:-(valence_movie_t/2)]
			else:
				valence_labels_movies[i] = valence_labels_movies[i][valence_movie_t/2:-(valence_movie_t/2) + 1]
		elif valence_movie_t == 2:
			valence_labels_movies[i] = valence_labels_movies[i][1:]

		if arousal_movie_t > 2:
			if arousal_movie_t % 2:
				arousal_labels_movies[i] = arousal_labels_movies[i][arousal_movie_t/2:-(arousal_movie_t/2)]
			else:
				arousal_labels_movies[i] = arousal_labels_movies[i][arousal_movie_t/2:-(arousal_movie_t/2) + 1]
		elif arousal_movie_t == 2:
			arousal_labels_movies[i] = arousal_labels_movies[i][1:]

	simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
		valence_labels_movies, arousal_labels_movies, threshold, valence_movie_t, arousal_movie_t, use_dev = False)
	# simple_cv(valence_elasticnet_regressors, arousal_elasticnet_regressors, valence_movie_matrices, arousal_movie_matrices, 
	# 	valence_labels_movies, arousal_labels_movies, threshold, valence_movie_t, arousal_movie_t)
