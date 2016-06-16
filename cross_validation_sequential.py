import numpy as np
import math
import sys
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.externals import joblib
from load import movies, load_output_movies
from matrix import join_matrices, join_vectors, rmse_matrix, extrapolate
from multiprocessing import Process, Array
from baseline_prediction import baseline_prediction
from sklearn.metrics import mean_squared_error
from threshold import threshold_n_features

def fold_training(valence_predictions, arousal_predictions, i,
	valence_regressors, arousal_regressors, 
	valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies, 
	n_test_matrices, n_train_matrices, n_valid_matrices, 
	n_valence_features, n_arousal_features):

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

	valence_test_matrix		= valence_test_matrix[:,n_valence_features:]
	valence_train_matrix	= valence_train_matrix[:,n_valence_features:]
	valence_valid_matrix	= valence_valid_matrix[:,n_valence_features:]
	arousal_test_matrix		= arousal_test_matrix[:,n_arousal_features:]
	arousal_train_matrix	= arousal_train_matrix[:,n_arousal_features:]
	arousal_valid_matrix	= arousal_valid_matrix[:,n_arousal_features:]

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

	print "Fold %d Best Valence Model %s" % (i, type(valence_model).__name__)
	print "Fold %d Best Arousal Model %s" % (i, type(arousal_model).__name__)

	print "Fold %d Finding predictions" % i
	valence_test_predictions = valence_model.predict(valence_test_matrix)
	arousal_test_predictions = arousal_model.predict(arousal_test_matrix)
	print "Fold %d Valence rmse = %f coeff %f" % (i, math.sqrt(mean_squared_error(valence_test_predictions, 
		valence_test_labels)), np.corrcoef(valence_test_predictions, valence_test_labels)[0][1])
	print "Fold %d Arousal rmse = %f coeff %f" % (i, math.sqrt(mean_squared_error(arousal_test_predictions, 
		arousal_test_labels)), np.corrcoef(arousal_test_predictions, arousal_test_labels)[0][1])
	print "Fold %d Done" % i
	return valence_test_predictions, arousal_test_predictions

def simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies, threshold, valence_movie_t, arousal_movie_t):
	n_train_matrices = 21
	n_valid_matrices = 6
	n_test_matrices = 3
	valence_labels = join_vectors(valence_labels_movies)
	arousal_labels = join_vectors(arousal_labels_movies)
	print len(valence_labels), len(arousal_labels)
	processes = []
	n_valence_features, n_arousal_features = threshold_n_features(threshold, valence_movie_t, arousal_movie_t)
	valence_predictions, arousal_predictions = np.array([], dtype = 'float'), np.array([], dtype = 'float')
	for i in range(0, 10):
		valence_test_predictions, arousal_test_predictions = fold_training(valence_predictions, arousal_predictions, i, 
			valence_regressors, arousal_regressors, 
			valence_movie_matrices, arousal_movie_matrices, 
			valence_labels_movies, arousal_labels_movies, 
			n_test_matrices, n_train_matrices, n_valid_matrices, 
			n_valence_features, n_arousal_features)
		valence_predictions = np.append(valence_predictions, valence_test_predictions)
		arousal_predictions = np.append(arousal_predictions, arousal_test_predictions)

	print math.sqrt(mean_squared_error(valence_labels, valence_predictions)), np.corrcoef(valence_labels, 
			valence_predictions)[0][1]
	print math.sqrt(mean_squared_error(arousal_labels, arousal_predictions)), np.corrcoef(arousal_labels, 
			arousal_predictions)[0][1]

if __name__ == '__main__':
	valence_regressors = [
						LinearRegression(), 
						ElasticNetCV(max_iter = 10000), 
						BayesianRidge(), 
						ElasticNet(), 
						DecisionTreeRegressor(max_depth = 2)
						]
	arousal_regressors = [
						LinearRegression(), 
						ElasticNetCV(max_iter = 10000), 
						BayesianRidge(), 
						ElasticNet(), 
						DecisionTreeRegressor(max_depth = 2)
						]
	valence_movie_t = int(sys.argv[1])
	arousal_movie_t = int(sys.argv[2])
	threshold = float(sys.argv[3])
	print valence_movie_t, arousal_movie_t, threshold
	arousal_movie_matrices = []
	valence_movie_matrices = []
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	for i, movie in enumerate(movies):
		n_annotations = min(len(arousal_labels_movies[i]), len(valence_labels_movies[i]))
		arousal_movie_matrices.append(np.load("../movie_matrices_movie_features/%s_arousal_%d.npy" % (movie, arousal_movie_t)))
		valence_movie_matrices.append(np.load("../movie_matrices_movie_features/%s_valence_%d.npy" % (movie, valence_movie_t)))
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
		valence_labels_movies, arousal_labels_movies, threshold, valence_movie_t, arousal_movie_t)

