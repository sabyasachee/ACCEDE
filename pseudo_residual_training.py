import sys
import numpy as np
import math
import copy
from load import movies, load_output_movies
from baseline_prediction import baseline_prediction
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
# from cross_validation_parallel import simple_cv
from cross_validation_matlab import simple_cv
from threshold import feature_selection
from best_shift import best_shift
from matrix import extrapolate
from MatlabRidge import MatlabRidge
import matplotlib.pyplot as plt

if __name__ == '__main__':
	valence_t = 10
	arousal_t = 10
	valence_movie_t = 3
	arousal_movie_t = 1
	threshold = 0.18
	best_valence_shifts = [0]
	best_arousal_shifts = [15]
	valence_shifts = [0,1,2,3,4,10]
	arousal_shifts = [9,13,14,15,16,17,25]
	smooths = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
	valence_smooths = range(len(movies))
	arousal_smooths = range(len(movies))
	try:
		valence_t = int(sys.argv[1])
		arousal_t = int(sys.argv[2])
		valence_movie_t = int(sys.argv[3])
		arousal_movie_t = int(sys.argv[4])
		threshold = float(sys.argv[5])
	except IndexError:
		print 'Using Default values'
		pass
	print 'window length for movies for prediction using video-clips model valence = %d arousal = %d' % \
		(valence_t, arousal_t)
	print 'window length for movies for training on pseudo residuals valence = %d arousal = %d' % \
		(valence_movie_t, arousal_movie_t)
	print 'threshold of correlation for feature selection = %f' % threshold
	# print 'Valence H0 smoothing window length %d' % valence_smooth
	# print 'Arousal H0 smoothing window length %d' % arousal_smooth
	
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	valence_movie_matrices, arousal_movie_matrices = [], []

	for movie in movies:
		valence_movie_matrices.append(np.load('../movie_matrices/%s_valence_%d.npy' % (movie, 
			valence_movie_t)))
		arousal_movie_matrices.append(np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, 
			arousal_movie_t)))

	valence_model, arousal_model = joblib.load('valence_model_ElasticNetCV.pkl'), \
		joblib.load('arousal_model_ElasticNetCV.pkl')
	valence_predictions_movies, arousal_predictions_movies = baseline_prediction(valence_t, arousal_t, 
		valence_model, arousal_model, valence_labels_movies, arousal_labels_movies)
	smoothened_valence_predictions_movies, smoothened_arousal_predictions_movies = range(len(movies)), \
		range(len(movies))

	for i in range(0,len(movies)):
		if valence_movie_t > 2:
			if valence_movie_t % 2:
				valence_predictions_movies[i] = valence_predictions_movies[i][valence_movie_t/2: \
					-(valence_movie_t/2)]
				valence_labels_movies[i] = valence_labels_movies[i][valence_movie_t/2:-(valence_movie_t/2)]
			else:
				valence_predictions_movies[i] = valence_predictions_movies[i][valence_movie_t/2: \
					-(valence_movie_t/2) + 1]
				valence_labels_movies[i] = valence_labels_movies[i][valence_movie_t/2:-(valence_movie_t/2) + 1]
		elif valence_movie_t == 2:
			valence_predictions_movies[i] = valence_predictions_movies[i][1:]
			valence_labels_movies[i] = valence_labels_movies[i][1:]

		if arousal_movie_t > 2:
			if arousal_movie_t % 2:
				arousal_predictions_movies[i] = arousal_predictions_movies[i][arousal_movie_t/2:\
					-(arousal_movie_t/2)]
				arousal_labels_movies[i] = arousal_labels_movies[i][arousal_movie_t/2:-(arousal_movie_t/2)]
			else:
				arousal_predictions_movies[i] = arousal_predictions_movies[i][arousal_movie_t/2:\
					-(arousal_movie_t/2) + 1]
				arousal_labels_movies[i] = arousal_labels_movies[i][arousal_movie_t/2:-(arousal_movie_t/2) + 1]
		elif arousal_movie_t == 2:
			arousal_predictions_movies[i] = arousal_predictions_movies[i][1:]
			arousal_labels_movies[i] = arousal_labels_movies[i][1:]

	print 'Applying valence smoothing'
	for i in range(0,len(movies)):
		rest_valence_labels_movies = valence_labels_movies[:i] + valence_labels_movies[i+1:]
		rest_valence_predictions_movies = valence_predictions_movies[:i] + valence_predictions_movies[i+1:]
		rest_valence_labels = np.hstack(tuple(rest_valence_labels_movies))
		rest_valence_predictions = np.hstack(tuple(rest_valence_predictions_movies))
		max_correlation = None
		valence_smooth = None
		for smooth in smooths:
			smoothenened_rest_valence_predictions = np.convolve(rest_valence_predictions, 
				np.ones((smooth,))/smooth, mode = 'same')
			correlation = np.corrcoef(smoothenened_rest_valence_predictions, rest_valence_labels)[0][1]
			if max_correlation is None or correlation > max_correlation:
				valence_smooth = smooth
				max_correlation = correlation
		valence_smooths[i] = valence_smooth

	for i in range(0,len(movies)):
		valence_smooths[i] = min(valence_smooths[i], len(valence_predictions_movies[i]))
		valence_predictions_movies[i] = np.convolve(valence_predictions_movies[i], 
			np.ones((valence_smooths[i],))/valence_smooths[i], mode = 'same')

	print 'Applying arousal smoothing'
	for i in range(0,len(movies)):
		rest_arousal_labels_movies = arousal_labels_movies[:i] + arousal_labels_movies[i+1:]
		rest_arousal_predictions_movies = arousal_predictions_movies[:i] + arousal_predictions_movies[i+1:]
		rest_arousal_labels = np.hstack(tuple(rest_arousal_labels_movies))
		rest_arousal_predictions = np.hstack(tuple(rest_arousal_predictions_movies))
		max_correlation = None
		arousal_smooth = None
		for smooth in smooths:
			smoothenened_rest_arousal_predictions = np.convolve(rest_arousal_predictions, 
				np.ones((smooth,))/smooth, mode = 'same')
			correlation = np.corrcoef(smoothenened_rest_arousal_predictions, rest_arousal_labels)[0][1]
			if max_correlation is None or correlation > max_correlation:
				arousal_smooth = smooth
				max_correlation = correlation
		arousal_smooths[i] = arousal_smooth

	for i in range(0,len(movies)):
		arousal_smooths[i] = min(arousal_smooths[i], len(arousal_predictions_movies[i]))
		arousal_predictions_movies[i] = np.convolve(arousal_predictions_movies[i], 
			np.ones((arousal_smooths[i],))/arousal_smooths[i], mode = 'same')	


	print 'Baseline performance:'
	valence_predictions = np.hstack(tuple(valence_predictions_movies))
	arousal_predictions = np.hstack(tuple(arousal_predictions_movies))
	valence_labels = np.hstack(tuple(valence_labels_movies))
	arousal_labels = np.hstack(tuple(arousal_labels_movies))
	print 'Valence rmse = %0.3f correlation = %0.3f' % (math.sqrt(mean_squared_error(valence_predictions, 
		valence_labels)), np.corrcoef(valence_predictions, valence_labels)[0][1])
	print 'Arousal rmse = %0.3f correlation = %0.3f' % (math.sqrt(mean_squared_error(arousal_predictions, 
		arousal_labels)), np.corrcoef(arousal_predictions, arousal_labels)[0][1])

	for k in range(10):
		print '**************************************************************'
		print 'iteration', k + 1

		shifted_valence_movie_matrices, shifted_arousal_movie_matrices = range(len(movies)), range(len(movies))
		diff_valence_labels_movies, diff_arousal_labels_movies = range(len(movies)), range(len(movies))
		shifted_valence_labels_movies, shifted_arousal_labels_movies = range(len(movies)), range(len(movies))
		extrapolated_valence_predictions_movies, extrapolated_arousal_predictions_movies = range(len(movies)), \
			range(len(movies))
		for i in range(len(movies)):
			diff_valence_labels_movies[i] = valence_labels_movies[i] - valence_predictions_movies[i]
			diff_arousal_labels_movies[i] = arousal_labels_movies[i] - arousal_predictions_movies[i]

		valence_regressors = [
							ElasticNetCV(max_iter = 100000),
							Ridge(alpha = 10.),
							Ridge(alpha = 100.),
							Ridge(alpha = 1000.),
							Ridge(alpha = 10000.)
							# ElasticNet(max_iter = 100000),
							# LinearRegression(),
							# BayesianRidge()
							]
		arousal_regressors = [
							ElasticNetCV(max_iter = 100000),
							Ridge(alpha = 10.),
							Ridge(alpha = 100.),
							Ridge(alpha = 1000.),
							Ridge(alpha = 10000.)
							# ElasticNet(max_iter = 100000),
							# LinearRegression(),
							# BayesianRidge()
							]
		
		print 'Feature Selection...'
		filtered_valence_movie_matrices, filtered_arousal_movie_matrices = feature_selection(
			valence_movie_matrices, arousal_movie_matrices, 
			diff_valence_labels_movies, diff_arousal_labels_movies, threshold)
		print filtered_valence_movie_matrices[0].shape, filtered_arousal_movie_matrices[0].shape
		
		print 'Finding best shift...'
		if k:
			valence_shift, arousal_shift = best_shift(filtered_valence_movie_matrices, 
				filtered_arousal_movie_matrices, 
				diff_valence_labels_movies, diff_arousal_labels_movies, 
				valence_shifts, arousal_shifts)
			best_valence_shifts.append(valence_shift)
			best_arousal_shifts.append(arousal_shift)
		else:
			valence_shift = best_valence_shifts[0]
			arousal_shift = best_arousal_shifts[0]

		print 'Shifting ...'
		if valence_shift:
			for i in range(len(movies)):
				shifted_valence_movie_matrices[i] = filtered_valence_movie_matrices[i][:-valence_shift,:]
				shifted_valence_labels_movies[i] = diff_valence_labels_movies[i][valence_shift:]
		else:
			shifted_valence_movie_matrices = filtered_valence_movie_matrices
			shifted_valence_labels_movies = diff_valence_labels_movies
		if arousal_shift:
			for i in range(len(movies)):
				shifted_arousal_movie_matrices[i] = filtered_arousal_movie_matrices[i][:-arousal_shift,:]
				shifted_arousal_labels_movies[i] = diff_arousal_labels_movies[i][arousal_shift:]
		else:
			shifted_arousal_movie_matrices = filtered_arousal_movie_matrices
			shifted_arousal_labels_movies = diff_arousal_labels_movies

		print 'Cross validation...'
		shifted_valence_predictions_movies, shifted_arousal_predictions_movies = simple_cv(k, 
			shifted_valence_movie_matrices, shifted_arousal_movie_matrices, 
			shifted_valence_labels_movies, shifted_arousal_labels_movies, n_folds = 30, use_dev = True) 

		shifted_valence_predictions = np.hstack(tuple(shifted_valence_predictions_movies))
		shifted_arousal_predictions = np.hstack(tuple(shifted_arousal_predictions_movies))
		shifted_valence_labels = np.hstack(tuple(shifted_valence_labels_movies))
		shifted_arousal_labels = np.hstack(tuple(shifted_arousal_labels_movies))

		print 'iteration', k + 1, 'performance on pseudo residuals:'
		print 'Valence rmse = %0.3f correlation = %0.3f' % (math.sqrt(mean_squared_error(
			shifted_valence_predictions, shifted_valence_labels)), 
			np.corrcoef(shifted_valence_predictions, shifted_valence_labels)[0][1])
		print 'Arousal rmse = %0.3f correlation = %0.3f' % (math.sqrt(mean_squared_error(
			shifted_arousal_predictions, shifted_arousal_labels)), 
			np.corrcoef(shifted_arousal_predictions, shifted_arousal_labels)[0][1])

		print 'Extrapolating predictions'
		for i in range(len(movies)):
			extrapolated_valence_predictions_movies[i] = extrapolate(len(valence_predictions_movies[i])
				, shifted_valence_predictions_movies[i], type = 2)
			extrapolated_arousal_predictions_movies[i] = extrapolate(len(arousal_predictions_movies[i])
				, shifted_arousal_predictions_movies[i], type = 2)

		# print 'Applying smoothing'
		# for i in range(len(movies)):
		# 	smooth_l = min(len(extrapolated_valence_predictions_movies[i]), valence_smooth)
		# 	extrapolated_valence_predictions_movies[i] = np.convolve(extrapolated_valence_predictions_movies[i], 
		# 		np.ones((smooth_l,))/smooth_l, mode = 'same')
		# 	smooth_l = min(len(extrapolated_arousal_predictions_movies[i]), arousal_smooth)
		# 	extrapolated_arousal_predictions_movies[i] = np.convolve(extrapolated_arousal_predictions_movies[i], 
		# 		np.ones((smooth_l,))/smooth_l, mode = 'same')

		
		print 'Applying valence smoothing'
		for i in range(len(movies)):
			rest_valence_labels_movies = valence_labels_movies[:i] + valence_labels_movies[i+1:]
			rest_valence_predictions_movies = extrapolated_valence_predictions_movies[:i] + \
				extrapolated_valence_predictions_movies[i+1:]
			rest_valence_labels = np.hstack(tuple(rest_valence_labels_movies))
			rest_valence_predictions = np.hstack(tuple(rest_valence_predictions_movies))
			max_correlation = None
			valence_smooth = None
			for smooth in smooths:
				smoothenened_rest_valence_predictions = np.convolve(rest_valence_predictions, 
					np.ones((smooth,))/smooth, mode = 'same')
				correlation = np.corrcoef(smoothenened_rest_valence_predictions, rest_valence_labels)[0][1]
				if max_correlation is None or correlation > max_correlation:
					valence_smooth = smooth
					max_correlation = correlation
			valence_smooths[i] = valence_smooth

		for i in range(len(movies)):
			valence_smooths[i] = min(valence_smooths[i], len(extrapolated_valence_predictions_movies[i]))
			extrapolated_valence_predictions_movies[i] = np.convolve(extrapolated_valence_predictions_movies[i], 
				np.ones((valence_smooths[i],))/valence_smooths[i], mode = 'same')

		print 'Applying arousal smoothing'
		for i in range(len(movies)):
			rest_arousal_labels_movies = arousal_labels_movies[:i] + arousal_labels_movies[i+1:]
			rest_arousal_predictions_movies = extrapolated_arousal_predictions_movies[:i] + \
				extrapolated_arousal_predictions_movies[i+1:]
			rest_arousal_labels = np.hstack(tuple(rest_arousal_labels_movies))
			rest_arousal_predictions = np.hstack(tuple(rest_arousal_predictions_movies))
			max_correlation = None
			arousal_smooth = None
			for smooth in smooths:
				smoothenened_rest_arousal_predictions = np.convolve(rest_arousal_predictions, 
					np.ones((smooth,))/smooth, mode = 'same')
				correlation = np.corrcoef(smoothenened_rest_arousal_predictions, rest_arousal_labels)[0][1]
				if max_correlation is None or correlation > max_correlation:
					arousal_smooth = smooth
					max_correlation = correlation
			arousal_smooths[i] = arousal_smooth

		for i in range(len(movies)):
			arousal_smooths[i] = min(arousal_smooths[i], len(extrapolated_arousal_predictions_movies[i]))
			extrapolated_arousal_predictions_movies[i] = np.convolve(extrapolated_arousal_predictions_movies[i], 
				np.ones((arousal_smooths[i],))/arousal_smooths[i], mode = 'same')


		print 'Adding to baseline'
		for i in range(len(movies)):
			valence_predictions_movies[i] += extrapolated_valence_predictions_movies[i]
			arousal_predictions_movies[i] += extrapolated_arousal_predictions_movies[i]
		valence_predictions = np.hstack(tuple(valence_predictions_movies))
		arousal_predictions = np.hstack(tuple(arousal_predictions_movies))
		valence_labels = np.hstack(tuple(valence_labels_movies))
		arousal_labels = np.hstack(tuple(arousal_labels_movies))
		print 'iteration', k + 1, 'new overall performance'
		print 'Valence rmse = %0.3f correlation = %0.3f' % (math.sqrt(mean_squared_error(valence_predictions, 
			valence_labels)), np.corrcoef(valence_predictions, valence_labels)[0][1])
		print 'Arousal rmse = %0.3f correlation = %0.3f' % (math.sqrt(mean_squared_error(arousal_predictions, 
			arousal_labels)), np.corrcoef(arousal_predictions, arousal_labels)[0][1])