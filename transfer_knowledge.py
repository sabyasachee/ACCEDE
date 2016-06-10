import numpy
import math
from sklearn.externals import joblib
import load
import utils
import matrix
import matplotlib.pyplot as plt

def predict(matrix, model):
	matrix = numpy.array(matrix, dtype = 'float')
	matrix = matrix.transpose()
	return numpy.array(model.predict(matrix), dtype = 'float')

def scale(row):
	return (row - 1)/2 - 1

def transfer_knowledge(valence_models, arousal_models, t = 10):
	# load labels and movies
	valence_labels, arousal_labels = load.load_output_movies()
	movies = load.movies
	movies_matrix = []
	for movie in movies:
		movies_matrix.append(load.load_input_movie(movie))
	feature_names = load.feature_names
	valence_header = []
	arousal_header = []
	for i, feature_name in enumerate(feature_names):
		for j in range(0, 9):
			valence_header.append((feature_name, j))
			arousal_header.append((feature_name, j))
	_, _, valence_correlations, arousal_correlations = load.load_output()

	# sort valence_header and arousal_header
	valence_header = sorted(valence_header, key = lambda valence_tuple: valence_correlations[valence_tuple[0]][valence_tuple[1]], reverse = True)
	arousal_header = sorted(arousal_header, key = lambda arousal_tuple: arousal_correlations[arousal_tuple[0]][arousal_tuple[1]], reverse = True)

	# iterate through each movie_matrix
	# get the valence and arousal labels
	# for each feature_vector, calculate statistics on t second window shifted by one second every step
	# get the new matrix, n_features * (statistics * duration)
	# sort the columns according to headers
	# make the predictions
	# scale the predictions

	for i, movie_matrix in enumerate(movies_matrix):
		print movies[i]
		n_annotations = min(len(arousal_labels[i]), len(valence_labels[i]))
		arousal_labels[i] = arousal_labels[i][:n_annotations]
		valence_labels[i] = valence_labels[i][:n_annotations]
		transformed_movie_matrix = matrix.window_matrix(movie_matrix, t, n_annotations)

		arousal_movie_matrix = matrix.sort_matrix(transformed_movie_matrix, arousal_correlations)
		valence_movie_matrix = matrix.sort_matrix(transformed_movie_matrix, valence_correlations)
		
		if t % 2:
			a_labels = numpy.array(arousal_labels[i][t/2:-t/2], dtype = 'float')
			v_labels = numpy.array(valence_labels[i][t/2:-t/2], dtype = 'float')
		else:
			a_labels = numpy.array(arousal_labels[i][t/2:-t/2 + 1], dtype = 'float')
			v_labels = numpy.array(valence_labels[i][t/2:-t/2 + 1], dtype = 'float')

		arousal_range = a_labels.max() - a_labels.min()
		valence_range = v_labels.max() - v_labels.min()

		max_valence_coeff = None
		best_valence_rmse = None
		best_valence_model = None

		max_arousal_coeff = None
		best_arousal_rmse = None
		best_arousal_model = None

		for arousal_model, valence_model in zip(arousal_models, valence_models):
			arousal_predictions = predict(arousal_movie_matrix, arousal_model)
			arousal_predictions = scale(arousal_predictions)

			valence_predictions = predict(valence_movie_matrix, valence_model)
			valence_predictions = scale(valence_predictions)

			a_difference = (a_labels - arousal_predictions)*(a_labels - arousal_predictions)
			v_difference = (v_labels - valence_predictions)*(v_labels - valence_predictions)

			rmse = math.sqrt(a_difference.mean())
			coeff = numpy.corrcoef(a_labels, arousal_predictions)[0][1]
			if max_arousal_coeff is None or max_arousal_coeff < coeff:
				max_arousal_coeff = coeff
				best_arousal_rmse = rmse
				best_arousal_model = arousal_model

			rmse = math.sqrt(v_difference.mean())
			coeff = numpy.corrcoef(v_labels, valence_predictions)[0][1]
			if max_valence_coeff is None or max_valence_coeff < coeff:
				max_valence_coeff = coeff
				best_valence_rmse = rmse
				best_valence_model = valence_model

		print "\t Valence model = %s rmse = %f coeff = %f range = %f" % (type(best_valence_model).__name__, 
			best_valence_rmse, max_valence_coeff, valence_range)
		print "\t Arousal model = %s rmse = %f coeff = %f range = %f" % (type(best_arousal_model).__name__, 
			best_arousal_rmse, max_arousal_coeff, arousal_range)


Bayesian_valence_model = joblib.load('valence_model_Bayesian.pkl')
Bayesian_arousal_model = joblib.load('arousal_model_Bayesian.pkl')

ElasticNet_valence_model = joblib.load('valence_model_ElasticNet.pkl')
ElasticNet_arousal_model = joblib.load('arousal_model_ElasticNet.pkl')

ElasticNetCV_valence_model = joblib.load('valence_model_ElasticNetCV.pkl')
ElasticNetCV_arousal_model = joblib.load('arousal_model_ElasticNetCV.pkl')

DecisionTree_valence_model = joblib.load('valence_model_DecisionTree.pkl')
DecisionTree_arousal_model = joblib.load('arousal_model_DecisionTree.pkl')

valence_models = [Bayesian_valence_model, ElasticNet_valence_model, ElasticNetCV_valence_model, 
				DecisionTree_valence_model]
arousal_models = [Bayesian_arousal_model, ElasticNet_arousal_model, ElasticNetCV_arousal_model, 
				DecisionTree_arousal_model]
transfer_knowledge(valence_models, arousal_models)
