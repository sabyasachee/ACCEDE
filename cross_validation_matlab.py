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
from MatlabRidge import MatlabRidge
from MatlabLasso import MatlabLasso
from EvalModel import WekaSVR
from sklearn.svm import LinearSVR
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, GaussianLayer, SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import warnings

def fold_training(core, valence_predictions, arousal_predictions,
	start_fold, end_fold, 
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
		# print "Fold %d valence filling %d to %d" % (i, valence_offset, valence_offset 
			# + len(valence_test_labels) - 1)
		labels_movies = arousal_labels_movies[:i*n_test_matrices]
		for labels_movie in labels_movies:
			arousal_offset += len(labels_movie)	
		# print "Fold %d arousal filling %d to %d" % (i, arousal_offset, arousal_offset 
			# + len(arousal_test_labels) - 1)		

		valence_rmse = None
		arousal_rmse = None
		valence_regressor = None
		arousal_regressor = None

		# Linear Regression
		vlr = LinearRegression()
		vlr.fit(valence_train_matrix, valence_train_labels)
		alr = LinearRegression()
		alr.fit(arousal_train_matrix, arousal_train_labels)
		valence_regressor = 'Linear'
		arousal_regressor = 'Linear'

		# # Ridge Regression
		# valence_ridge_coeffs = None
		# arousal_ridge_coeffs = None
		# valence_ridge_alpha = None
		# arousal_ridge_alpha = None
		# ridge_alphas = [10., 100., 1000., 10000., 100000.]
		# # ridge_alphas = [0., 0.]

		# coeffs_matrix = MatlabRidge(valence_train_matrix, valence_train_labels, ridge_alphas, core)
		# print coeffs_matrix.shape, valence_train_matrix.shape, valence_train_labels.shape
		# augmented_valence_valid_matrix = np.insert(valence_valid_matrix, 0, 
		# 	np.ones(len(valence_valid_matrix)), axis = 1)
		# for j,coeffs in enumerate(coeffs_matrix.T):
		# 	valence_valid_predictions = np.dot(augmented_valence_valid_matrix, coeffs)
		# 	rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
		# 	if valence_rmse is None or valence_rmse > rmse:
		# 		valence_rmse = rmse
		# 		valence_ridge_alpha = ridge_alphas[j]
		# 		valence_ridge_coeffs = coeffs
		# 		valence_regressor = 'Ridge'

		# coeffs_matrix = MatlabRidge(arousal_train_matrix, arousal_train_labels, ridge_alphas, core)
		# print coeffs_matrix.shape, arousal_train_matrix.shape, arousal_train_labels.shape
		# augmented_arousal_valid_matrix = np.insert(arousal_valid_matrix, 0, 
		# 	np.ones(len(arousal_valid_matrix)), axis = 1)
		# for j,coeffs in enumerate(coeffs_matrix.T):
		# 	arousal_valid_predictions = np.dot(augmented_arousal_valid_matrix, coeffs)
		# 	rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
		# 	if arousal_rmse is None or arousal_rmse > rmse:
		# 		arousal_rmse = rmse
		# 		arousal_ridge_alpha = ridge_alphas[j]
		# 		arousal_ridge_coeffs = coeffs
		# 		arousal_regressor = 'Ridge'

		# Elastic Net Regression
		# valence_elasticnet_coeffs, valence_elasticnet_bias = None, None
		# arousal_elasticnet_coeffs, arousal_elasticnet_bias = None, None
		# valence_elasticnet_alpha, valence_elasticnet_lambda = None, None
		# arousal_elasticnet_alpha, arousal_elasticnet_lambda = None, None
		# elasticnet_alphas = [.001, .01, .1, .5, .7]
		# elasticnet_lambdas = [.1, 10, 100, 1000, 10000, 100000]

		# for j, alpha in enumerate(elasticnet_alphas):
		# 	coeffs_matrix, biases = MatlabLasso(valence_train_matrix, valence_train_labels, alpha, 
		# 		elasticnet_lambdas, core)
		# 	# print coeffs_matrix.shape, valence_train_matrix.shape, valence_train_labels.shape
		# 	for k, coeffs in enumerate(coeffs_matrix.T):
		# 		valence_valid_predictions = np.reshape(np.dot(valence_valid_matrix, coeffs), 
		# 			(valence_valid_labels.shape[0],)) + np.repeat(biases[k], len(valence_valid_labels))
		# 		rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
		# 		if valence_rmse is None or valence_rmse > rmse:
		# 			valence_rmse = rmse
		# 			valence_elasticnet_alpha = alpha
		# 			valence_elasticnet_lambda = elasticnet_lambdas[k]
		# 			valence_elasticnet_coeffs = coeffs
		# 			valence_elasticnet_bias = biases[k]
		# 			valence_regressor = 'ElasticNet'

		# for j, alpha in enumerate(elasticnet_alphas):
		# 	coeffs_matrix, biases = MatlabLasso(arousal_train_matrix, arousal_train_labels, alpha, 
		# 		elasticnet_lambdas, core)
		# 	# print coeffs_matrix.shape, arousal_train_matrix.shape, arousal_train_labels.shape
		# 	for k, coeffs in enumerate(coeffs_matrix.T):
		# 		arousal_valid_predictions = np.reshape(np.dot(arousal_valid_matrix, coeffs), 
		# 			(arousal_valid_labels.shape[0],)) + np.repeat(biases[k], len(arousal_valid_labels))
		# 		rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
		# 		if arousal_rmse is None or arousal_rmse > rmse:
		# 			arousal_rmse = rmse
		# 			arousal_elasticnet_alpha = alpha
		# 			arousal_elasticnet_lambda = elasticnet_lambdas[k]
		# 			arousal_elasticnet_coeffs = coeffs
		# 			arousal_elasticnet_bias = biases[k]
		# 			arousal_regressor = 'ElasticNet'

		# Weka SVR Regression
		# valence_svr_model, arousal_svr_model = None, None
		# valence_svr_c, arousal_svr_c = None, None
		# c_vals = [.001, .01, .1, 1., 10., 100., 1000.]
		# c_vals.reverse()
		# for c in c_vals:
		# 	print 'Fold %d working with c %f' % (i, c)
		# 	weka = WekaSVR(remove_model_file = False, 
		# 		force_model_filename = 'WekaSVR_valence_c_%d_fold_%d' % (c, i))
		# 	weka.fit(valence_train_matrix, valence_train_labels, c_val = c)
		# 	valence_valid_predictions = np.array(weka.predict(valence_valid_matrix), dtype = 'float')
		# 	rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
		# 	if valence_rmse is None or valence_rmse > rmse:
		# 		valence_rmse = rmse
		# 		valence_svr_c = c
		# 		valence_regressor = 'SVR'

		# for c in c_vals:
		# 	print 'Fold %d working with c %f' % (i, c)
		# 	weka = WekaSVR(remove_model_file = False, 
		# 		force_model_filename = 'WekaSVR_arousal_c_%d_fold_%d' % (c, i))
		# 	weka.fit(arousal_train_matrix, arousal_train_labels, c_val = c)
		# 	arousal_valid_predictions = np.array(weka.predict(arousal_valid_matrix), dtype = 'float')
		# 	rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
		# 	if arousal_rmse is None or arousal_rmse > rmse:
		# 		arousal_rmse = rmse
		# 		arousal_svr_c = c
		# 		arousal_regressor = 'SVR'

		# Sklearn Linear SVR
		# valence_linearsvr_model, arousal_linearsvr_model = None, None
		# valence_linearsvr_c, arousal_linearsvr_c = None, None
		# c_vals = [.001, .0001, .000001]
		# for c in c_vals:
		# 	print 'Fold %d working with c %f' % (i, c)
		# 	regressor = LinearSVR(C = c)
		# 	regressor.fit(valence_train_matrix, valence_train_labels)
		# 	valence_valid_predictions = np.array(regressor.predict(valence_valid_matrix), dtype = 'float')
		# 	rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
		# 	if valence_rmse is None or valence_rmse > rmse:
		# 		valence_rmse = rmse
		# 		valence_linearsvr_c = c
		# 		valence_linearsvr_model = regressor
		# 		valence_regressor = 'LinearSVR'

		# for c in c_vals:
		# 	print 'Fold %d working with c %f' % (i, c)
		# 	regressor = LinearSVR(C = c)
		# 	regressor.fit(arousal_train_matrix, arousal_train_labels)
		# 	arousal_valid_predictions = np.array(regressor.predict(arousal_valid_matrix), dtype = 'float')
		# 	rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
		# 	if arousal_rmse is None or arousal_rmse > rmse:
		# 		arousal_rmse = rmse
		# 		arousal_linearsvr_c = c
		# 		arousal_linearsvr_model = regressor
		# 		arousal_regressor = 'LinearSVR'

		# # Neural Net 
		# valence_test_dataset = SupervisedDataSet(valence_test_matrix.shape[1], 1)
		# valence_train_dataset = SupervisedDataSet(valence_train_matrix.shape[1], 1)
		# if use_dev:
		# 	valence_valid_dataset = SupervisedDataSet(valence_valid_matrix.shape[1], 1)
		# valence_test_dataset.setField('input', valence_test_matrix)
		# valence_train_dataset.setField('input', valence_train_matrix)
		# valence_test_dataset.setField('target', np.zeros((valence_test_labels.shape[0],1)))
		# valence_train_dataset.setField('target', valence_train_labels.reshape(-1,1))
		# if use_dev:
		# 	valence_valid_dataset.setField('input', valence_valid_matrix)
		# 	valence_valid_dataset.setField('target', np.zeros((valence_valid_labels.shape[0],1)))

		# arousal_test_dataset = SupervisedDataSet(arousal_test_matrix.shape[1], 1)
		# arousal_train_dataset = SupervisedDataSet(arousal_train_matrix.shape[1], 1)
		# if use_dev:
		# 	arousal_valid_dataset = SupervisedDataSet(arousal_valid_matrix.shape[1], 1)
		# arousal_test_dataset.setField('input', arousal_test_matrix)
		# arousal_train_dataset.setField('input', arousal_train_matrix)
		# arousal_test_dataset.setField('target', np.zeros((arousal_test_labels.shape[0],1)))
		# arousal_train_dataset.setField('target', arousal_train_labels.reshape(-1,1))
		# if use_dev:
		# 	arousal_valid_dataset.setField('input', arousal_valid_matrix)
		# 	arousal_valid_dataset.setField('target', np.zeros((arousal_valid_labels.shape[0],1)))

		# n_neurons_list = [100, 200, 300, 400, 500, 600, 700]
		# layer_classes = [SigmoidLayer, TanhLayer]
		# epochs = 10

		# valence_n_neurons, arousal_n_neurons = None, None
		# valence_layer_class, arousal_layer_class = None, None
		# valence_net, arousal_net = None, None

		# for layer_class in layer_classes:
		# 	for n_neurons in n_neurons_list:
		# 		print 'Fold %d n_neurons %d layer_class %s' % (i, n_neurons, layer_class.__name__)
		# 		net = buildNetwork(valence_train_matrix.shape[1], n_neurons, 1, bias = True, 
		# 			hiddenclass = layer_class)
		# 		trainer = BackpropTrainer(net, valence_train_dataset, weightdecay = 0.01, learningrate = 0.005)
		# 		with warnings.catch_warnings():
		# 			warnings.filterwarnings('error')
		# 			try:
		# 				trainer.trainEpochs(epochs = epochs)
		# 			except Warning as e:
		# 				print 'Fold %d Runtime Warning encountered' % i
		# 				continue
		# 		valence_valid_predictions = net.activateOnDataset(valence_valid_dataset)
		# 		valence_valid_predictions = valence_valid_predictions.flatten()
		# 		rmse = math.sqrt(mean_squared_error(valence_valid_predictions, valence_valid_labels))
		# 		if valence_rmse is None or valence_rmse > rmse:
		# 			valence_rmse = rmse
		# 			valence_n_neurons = n_neurons
		# 			valence_layer_class = layer_class
		# 			valence_net = net.copy()
		# 			valence_regressor = 'NN'

		# for layer_class in layer_classes:
		# 	for n_neurons in n_neurons_list:
		# 		print 'Fold %d n_neurons %d layer_class %s' % (i, n_neurons, layer_class.__name__)
		# 		net = buildNetwork(arousal_train_matrix.shape[1], n_neurons, 1, bias = True, 
		# 			hiddenclass = layer_class)
		# 		trainer = BackpropTrainer(net, arousal_train_dataset, weightdecay = 0.01, learningrate = 0.005)
		# 		with warnings.catch_warnings():
		# 			warnings.filterwarnings('error')
		# 			try:
		# 				trainer.trainEpochs(epochs = epochs)
		# 			except Warning as e:
		# 				print 'Fold %d Runtime Warning encountered' % i
		# 				continue
		# 		arousal_valid_predictions = net.activateOnDataset(arousal_valid_dataset)
		# 		arousal_valid_predictions = arousal_valid_predictions.flatten()
		# 		rmse = math.sqrt(mean_squared_error(arousal_valid_predictions, arousal_valid_labels))
		# 		if arousal_rmse is None or arousal_rmse > rmse:
		# 			arousal_rmse = rmse
		# 			arousal_n_neurons = n_neurons
		# 			arousal_layer_class = layer_class
		# 			arousal_net = net.copy()
		# 			arousal_regressor = 'NN'	

		if valence_regressor == 'Ridge':
			print 'Fold %d best valence model Ridge alpha %f' % (i, valence_ridge_alpha)
			augmented_valence_test_matrix = np.insert(valence_test_matrix, 0, np.ones(len(valence_test_matrix)), 
			axis = 1)
			valence_test_predictions = np.dot(augmented_valence_test_matrix, valence_ridge_coeffs)
		 	valence_test_predictions = np.reshape(valence_test_predictions, (valence_test_predictions.shape[0],))
		if valence_regressor == 'ElasticNet':
			print 'Fold %d best valence model ElasticNet alpha %f lambda %f' % (i, valence_elasticnet_alpha, 
				valence_elasticnet_lambda)
			valence_test_predictions = np.reshape(np.dot(valence_test_matrix, valence_elasticnet_coeffs), 
					(valence_test_labels.shape[0],)) + np.repeat(valence_elasticnet_bias, len(valence_test_labels))
		if valence_regressor == 'SVR':
			print 'Fold %d best valence model SVR c %f' % (i, valence_svr_c)
			weka = WekaSVR(model_file = 'WekaSVR_valence_c_%d_fold_%d' % (valence_svr_c, i))
			valence_test_predictions = np.array(weka.predict(valence_test_matrix), dtype = 'float')
		if valence_regressor == 'LinearSVR':
			print 'Fold %d best valence model LinearSVR c %f' % (i, valence_linearsvr_c)
			valence_test_predictions = np.array(valence_linearsvr_model.predict(valence_test_matrix), dtype = 'float')
		if valence_regressor == 'NN':
			print 'Fold %d best valence model NN n_neurons %d layer_class %s' % (i, valence_n_neurons, 
				valence_layer_class.__name__)
			valence_test_predictions = valence_net.activateOnDataset(valence_test_dataset)
			valence_test_predictions = valence_test_predictions.flatten()
		if valence_regressor == 'Linear':
			print 'Fold %d best valence model Linear Regression' % i
			valence_test_predictions = vlr.predict(valence_test_matrix)

		if arousal_regressor == 'Ridge':
			print 'Fold %d best arousal model Ridge alpha %f' % (i, arousal_ridge_alpha)
			augmented_arousal_test_matrix = np.insert(arousal_test_matrix, 0, np.ones(len(arousal_test_matrix)), 
			axis = 1)
			arousal_test_predictions = np.dot(augmented_arousal_test_matrix, arousal_ridge_coeffs)
		 	arousal_test_predictions = np.reshape(arousal_test_predictions, (arousal_test_predictions.shape[0],))
		if arousal_regressor == 'ElasticNet':
			print 'Fold %d best arousal model ElasticNet alpha %f lambda %f' % (i, arousal_elasticnet_alpha, 
				arousal_elasticnet_lambda)
			arousal_test_predictions = np.reshape(np.dot(arousal_test_matrix, arousal_elasticnet_coeffs), 
					(arousal_test_labels.shape[0],)) + np.repeat(arousal_elasticnet_bias, len(arousal_test_labels))
		if arousal_regressor == 'SVR':
			print 'Fold %d best arousal model SVR c %f' % (i, arousal_svr_c)
			weka = WekaSVR(model_file = 'WekaSVR_arousal_c_%d_fold_%d' % (arousal_svr_c, i))
			arousal_test_predictions = np.array(weka.predict(arousal_test_matrix), dtype = 'float')
		if arousal_regressor == 'LinearSVR':
			print 'Fold %d best arousal model LinearSVR c %f' % (i, arousal_linearsvr_c)
			arousal_test_predictions = np.array(arousal_linearsvr_model.predict(arousal_test_matrix), dtype = 'float')
		if arousal_regressor == 'NN':
			print 'Fold %d best arousal model NN n_neurons %d layer_class %s' % (i, arousal_n_neurons, 
				arousal_layer_class.__name__)
			arousal_test_predictions = arousal_net.activateOnDataset(arousal_test_dataset)
			arousal_test_predictions = arousal_test_predictions.flatten()
		if arousal_regressor == 'Linear':
			print 'Fold %d best arousal model Linear Regression' % i
			arousal_test_predictions = alr.predict(arousal_test_matrix)
		
		valence_rmse = math.sqrt(mean_squared_error(valence_test_predictions, valence_test_labels))
		valence_correlation = np.corrcoef(valence_test_predictions, valence_test_labels)[0][1]
		arousal_rmse = math.sqrt(mean_squared_error(arousal_test_predictions, arousal_test_labels))
		arousal_correlation = np.corrcoef(arousal_test_predictions, arousal_test_labels)[0][1]
		print "Fold %d Valence rmse = %f coeff  = %f" % (i, valence_rmse, valence_correlation)
		print "Fold %d Arousal rmse = %f coeff  = %f" % (i, arousal_rmse, arousal_correlation)

		for j, valence_test_prediction in enumerate(valence_test_predictions):
			valence_predictions[valence_offset + j] = valence_test_prediction
		for j, arousal_test_prediction in enumerate(arousal_test_predictions):
			arousal_predictions[arousal_offset + j] = arousal_test_prediction

		print "Fold %d Done" % i
		# return valence_test_predictions, arousal_test_predictions
	print 'All done'
	return

def simple_cv(iteration, valence_movie_matrices, arousal_movie_matrices, valence_labels_movies, 
	arousal_labels_movies, n_folds = 10, use_dev = True):

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
	# valence_rmses, arousal_rmses = Array('d', n_movies), Array('d', n_movies)
	# valence_correlations, arousal_correlations = Array('d', n_movies), Array('d', n_movies)
	# valence_predictions = np.array([], dtype = 'float')
	# arousal_predictions = np.array([], dtype = 'float')
	processes = []
	n_cores = 5
	for i in range(0, n_cores):
		process = Process(target = fold_training, args = (i, valence_predictions, arousal_predictions, 
			(n_folds*i)/n_cores, (n_folds*(i + 1))/n_cores, 
			valence_movie_matrices, arousal_movie_matrices, 
			valence_labels_movies, arousal_labels_movies, 
			n_test_matrices, n_train_matrices, n_valid_matrices, 
			use_dev, ))
		processes.append(process)
		process.start()

	for j, process in enumerate(processes):
		process.join()
		print 'process', j, 'joined'
	print "All processes joined"
	# for i in range(n_folds):
	# 	valence_test_predictions, arousal_test_predictions = fold_training(i, 
	# 		valence_movie_matrices, arousal_movie_matrices, 
	# 		valence_labels_movies, arousal_labels_movies, 
	# 		n_test_matrices, n_train_matrices, n_valid_matrices, 
	# 		use_dev)
	# 	valence_predictions = np.hstack((valence_predictions, valence_test_predictions))
	# 	arousal_predictions = np.hstack((arousal_predictions, arousal_test_predictions))

	# with open('valence_iteration_%d.txt' % iteration, 'w') as f:
	# 	f.write('movie\trmse\tcorrelation\n')
	# 	for i in range(len(movies)):
	# 		f.write('%s\t%f\t%f\n' % (movies[i], valence_rmses[i], valence_correlations[i]))
	# with open('arousal_iteration_%d.txt' % iteration, 'w') as f:
	# 	f.write('movie\trmse\tcorrelation\n')
	# 	for i in range(len(movies)):
	# 		f.write('%s\t%f\t%f\n' % (movies[i], arousal_rmses[i], arousal_correlations[i]))

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
		valence_movie_matrices.append(np.load("../movie_matrices_movie_features/%s_valence_%d.npy" % 
			(movie, valence_movie_t)))
		arousal_movie_matrices.append(np.load("../movie_matrices_movie_features/%s_arousal_%d.npy" % 
			(movie, arousal_movie_t)))
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
	# simple_cv(valence_elasticnet_regressors, arousal_elasticnet_regressors, valence_movie_matrices, 
		# arousal_movie_matrices, 
	# 	valence_labels_movies, arousal_labels_movies, threshold, valence_movie_t, arousal_movie_t)
