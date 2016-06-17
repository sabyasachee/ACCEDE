import sys
import numpy as np
from load import movies, load_output_movies
from baseline_prediction import baseline_prediction
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from cross_validation_parallel import simple_cv
from threshold import feature_selection
from best_shift import best_shift

if __name__ == '__main__':
	valence_t = 10
	arousal_t = 10
	valence_movie_t = 3
	arousal_movie_t = 1
	threshold = 0.05
	valence_shift = 0
	arousal_shift = 15
	valence_shifts = [0,1,2,3,4,10]
	arousal_shifts = [9,13,14,15,16,17,25]
	try:
		valence_t = int(sys.argv[1])
		arousal_t = int(sys.argv[2])
		valence_movie_t = int(sys.argv[3])
		arousal_movie_t = int(sys.argv[4])
		threshold = float(sys.argv[5])
	except IndexError:
		print 'Using Default values'
		pass
	print 'window length for movies for prediction using video-clips model valence = %d arousal = %d' % (valence_t, arousal_t)
	print 'window length for movies for training on pseudo residuals valence = %d arousal = %d' % (valence_movie_t, arousal_movie_t)
	print 'threshold of correlation for feature selection = %f' % threshold
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	valence_movie_matrices, arousal_movie_matrices = [], []

	for movie in movies:
		valence_movie_matrices.append(np.load('../movie_matrices_movie_features/%s_valence_%d.npy' % (movie, 
			valence_movie_t)))
		arousal_movie_matrices.append(np.load('../movie_matrices_movie_features/%s_arousal_%d.npy' % (movie, 
			arousal_movie_t)))

	valence_baseline_model, arousal_baseline_model = joblib.load('valence_model_ElasticNetCV.pkl'), \
		joblib.load('arousal_model_ElasticNetCV.pkl')
	diff_valence_labels_movies, diff_arousal_labels_movies = baseline_prediction(valence_t, arousal_t, 
		valence_baseline_model, arousal_baseline_model, valence_labels_movies, arousal_labels_movies)
	print 'Pseudo residual errors obtained'

	for i in range(len(movies)):
		if valence_movie_t > 2:
			if valence_movie_t % 2:
				diff_valence_labels_movies[i] = diff_valence_labels_movies[i][valence_movie_t/2:-(valence_movie_t/2)]
			else:
				diff_valence_labels_movies[i] = diff_valence_labels_movies[i][valence_movie_t/2:-(valence_movie_t/2) + 1]
		elif valence_movie_t == 2:
			diff_valence_labels_movies[i] = diff_valence_labels_movies[i][1:]

		if arousal_movie_t > 2:
			if arousal_movie_t % 2:
				diff_arousal_labels_movies[i] = diff_arousal_labels_movies[i][arousal_movie_t/2:-(arousal_movie_t/2)]
			else:
				diff_arousal_labels_movies[i] = diff_arousal_labels_movies[i][arousal_movie_t/2:-(arousal_movie_t/2) + 1]
		elif arousal_movie_t == 2:
			diff_arousal_labels_movies[i] = diff_arousal_labels_movies[i][1:]

	for k in range(0, 5):
		print '**************************************************************'
		print 'iteration', k
		valence_regressors = [
							# ElasticNetCV(max_iter = 100000),
							# ElasticNet(max_iter = 100000),
							LinearRegression(),
							# BayesianRidge()
							]
		arousal_regressors = [
							# ElasticNetCV(max_iter = 100000),
							# ElasticNet(max_iter = 100000),
							LinearRegression(),
							# BayesianRidge()
							]
		print 'Feature Selection...'
		valence_movie_matrices, arousal_movie_matrices = feature_selection(valence_movie_matrices, 
			arousal_movie_matrices, diff_valence_labels_movies, diff_arousal_labels_movies, threshold)
		print 'Shifting...'
		if k:
			valence_shift, arousal_shift = best_shift(valence_movie_matrices, arousal_movie_matrices, diff_valence_labels_movies, 
					diff_arousal_labels_movies, valence_shifts, arousal_shifts)
		if valence_shift:
			for i in range(len(movies)):
				valence_movie_matrices[i] = valence_movie_matrices[i][:-valence_shift,:]
				diff_valence_labels_movies[i] = diff_valence_labels_movies[i][valence_shift:]
		if arousal_shift:
			for i in range(len(movies)):
				arousal_movie_matrices[i] = arousal_movie_matrices[i][:-arousal_shift,:]
				diff_arousal_labels_movies[i] = diff_arousal_labels_movies[i][arousal_shift:]
		for i in range(len(movies)):
			print valence_movie_matrices[i].shape, diff_valence_labels_movies[i].shape
			print arousal_movie_matrices[i].shape, diff_arousal_labels_movies[i].shape

		print 'Cross validation...'
		diff_valence_labels_movies, diff_arousal_labels_movies = simple_cv(valence_regressors, arousal_regressors, 
			valence_movie_matrices, arousal_movie_matrices, 
			diff_valence_labels_movies, diff_arousal_labels_movies, n_folds = 30, use_dev = False) 