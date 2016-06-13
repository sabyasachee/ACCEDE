import numpy as np
import math
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.externals import joblib
from load import movies, load_output_movies
from matrix import join_matrices, join_vectors, rmse_matrix, extrapolate
from multiprocessing import Process, Array
from baseline_prediction import baseline_prediction
from sklearn.metrics import mean_squared_error

def fold_training(valence_predictions, arousal_predictions, i,
	valence_regressors, arousal_regressors, 
	valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies, 
	n_test_matrices, n_train_matrices, n_valid_matrices):

	print "Fold %d" % i
	valence_test_matrices	= valence_movie_matrices[i*n_test_matrices: (i+1)*n_test_matrices]
	valence_test_labels		= valence_labels_movies[i*n_test_matrices: (i+1)*n_test_matrices]
	rest_valence_matrices	= valence_movie_matrices[:i*n_test_matrices] + \
		valence_movie_matrices[(i+1)*n_test_matrices:]
	rest_valence_labels		= valence_labels_movies[:i*n_test_matrices] + \
		valence_labels_movies[(i+1)*n_test_matrices:]
	valence_train_matrices	= rest_valence_matrices[:n_train_matrices]
	valence_train_labels	= rest_valence_labels[:n_train_matrices]
	valence_valid_matrices	= rest_valence_matrices[n_train_matrices:]
	valence_valid_labels	= rest_valence_labels[n_train_matrices:]

	arousal_test_matrices	= arousal_movie_matrices[i*n_test_matrices: (i+1)*n_test_matrices]
	arousal_test_labels		= arousal_labels_movies[i*n_test_matrices: (i+1)*n_test_matrices]
	rest_arousal_matrices	= arousal_movie_matrices[:i*n_test_matrices] + \
		arousal_movie_matrices[(i+1)*n_test_matrices:]
	rest_arousal_labels		= arousal_labels_movies[:i*n_test_matrices] + \
		arousal_labels_movies[(i+1)*n_test_matrices:]
	arousal_train_matrices	= rest_arousal_matrices[:n_train_matrices]
	arousal_train_labels	= rest_arousal_labels[:n_train_matrices]
	arousal_valid_matrices	= rest_arousal_matrices[n_train_matrices:]
	arousal_valid_labels	= rest_arousal_labels[n_train_matrices:]

	valence_test_matrix		= join_matrices(valence_test_matrices)  
	valence_test_labels		= join_vectors(valence_test_labels)
	valence_train_matrix	= join_matrices(valence_train_matrices)
	valence_train_labels	= join_vectors(valence_train_labels)
	valence_valid_matrix	= join_matrices(valence_valid_matrices)
	valence_valid_labels	= join_vectors(valence_valid_labels)

	arousal_test_matrix		= join_matrices(arousal_test_matrices)  
	arousal_test_labels		= join_vectors(arousal_test_labels)
	arousal_train_matrix	= join_matrices(arousal_train_matrices)
	arousal_train_labels	= join_vectors(arousal_train_labels)
	arousal_valid_matrix	= join_matrices(arousal_valid_matrices)
	arousal_valid_labels	= join_vectors(arousal_valid_labels)

	# print "Fold %d length of valence_labels %d, arousal_labels %d" % (i, len(valence_test_labels), 
	# 	len(arousal_test_labels))

	offset = 0
	labels_movies = valence_labels_movies[:i*n_test_matrices]
	for labels_movie in labels_movies:
		offset += len(labels_movie)	
	print "Fold %d filling %d to %d" % (i, offset, offset + len(valence_test_labels) - 1)

	valence_rmse = None
	arousal_rmse = None
	valence_model = None
	arousal_model = None
	for regressor in valence_regressors:
		print "\tFold %d Valence regressor %s" % (i, type(regressor).__name__)
		regressor.fit(valence_train_matrix, valence_train_labels)
		valence_valid_predictions = regressor.predict(valence_valid_matrix)
		rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
		if valence_rmse is None or valence_rmse > rmse:
			valence_rmse = rmse
			valence_model = regressor

	for regressor in arousal_regressors:
		print "\tFold %d Arousal regressor %s" % (i, type(regressor).__name__)
		regressor.fit(arousal_train_matrix, arousal_train_labels)
		arousal_valid_predictions = regressor.predict(arousal_valid_matrix)
		rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
		if arousal_rmse is None or arousal_rmse > rmse:
			arousal_rmse = rmse
			arousal_model = regressor

	print "Fold %d Finding predictions" % i
	valence_test_predictions = valence_model.predict(valence_test_matrix)
	arousal_test_predictions = arousal_model.predict(arousal_test_matrix)
	print "Fold %d Valence rmse = %f coeff %f" % (i, math.sqrt(mean_squared_error(valence_test_predictions, 
		valence_test_labels)), np.corrcoef(valence_test_predictions, valence_test_labels)[0][1])
	print "Fold %d Arousal rmse = %f coeff %f" % (i, math.sqrt(mean_squared_error(arousal_test_predictions, 
		arousal_test_labels)), np.corrcoef(arousal_test_predictions, arousal_test_labels)[0][1])
	for j, valence_test_prediction in enumerate(valence_test_predictions):
		valence_predictions[offset + j] = valence_test_prediction
	for j, arousal_test_prediction in enumerate(arousal_test_predictions):
		arousal_predictions[offset + j] = arousal_test_prediction
	print "Fold %d Done" % i
	return

def simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies):
	n_train_matrices = 21
	n_valid_matrices = 6
	n_test_matrices = 3
	valence_labels = join_vectors(valence_labels_movies)
	arousal_labels = join_vectors(arousal_labels_movies)
	valence_predictions = Array('d', len(valence_labels))
	arousal_predictions = Array('d', len(arousal_labels))
	print len(valence_labels), len(arousal_labels)
	processes = []
	for i in range(0, 10):
		process = Process(target = fold_training, args = (valence_predictions, arousal_predictions, i, 
			valence_regressors, arousal_regressors, 
			valence_movie_matrices, arousal_movie_matrices, 
			valence_labels_movies, arousal_labels_movies, 
			n_test_matrices, n_train_matrices, n_valid_matrices,))
		processes.append(process)
		process.start()

	for process in processes:
		process.join()
	print "All processes joined"

	valence_predictions = np.array(valence_predictions, dtype = 'float')
	arousal_predictions = np.array(arousal_predictions, dtype = 'float')
	print math.sqrt(mean_squared_error(valence_labels, valence_predictions)), np.corrcoef(valence_labels, 
			valence_predictions)[0][1]
	print math.sqrt(mean_squared_error(arousal_labels, arousal_predictions)), np.corrcoef(arousal_labels, 
			arousal_predictions)[0][1]

if __name__ == '__main__':
	# valence_regressors = [LinearRegression(), ElasticNetCV(max_iter = 10000), BayesianRidge(), ElasticNet(), 
	# 	DecisionTreeRegressor(max_depth = 2)]
	# arousal_regressors = [LinearRegression(), ElasticNetCV(max_iter = 10000), BayesianRidge(), ElasticNet(), 
	# 	DecisionTreeRegressor(max_depth = 2)]
	valence_regressors = [ElasticNetCV(max_iter = 10000)]
	arousal_regressors = [ElasticNetCV(max_iter = 10000)]
	arousal_movie_matrices = []
	valence_movie_matrices = []
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	valence_t = 10
	arousal_t = 10
	valence_model = joblib.load('valence_model_ElasticNetCV.pkl')
	arousal_model = joblib.load('arousal_model_ElasticNetCV.pkl')
	for i, movie in enumerate(movies):
		arousal_movie_matrices.append(np.load("../movie_matrices/%s_arousal_1.npy" % movie))
		valence_movie_matrices.append(np.load("../movie_matrices/%s_valence_1.npy" % movie))
		n_annotations = min(len(arousal_labels_movies[i]), len(valence_labels_movies[i]))
		arousal_labels_movies[i] = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')
		valence_labels_movies[i] = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')

	simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
		valence_labels_movies, arousal_labels_movies)

