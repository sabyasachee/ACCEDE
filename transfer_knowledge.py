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
	print matrix.shape
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
		n_annotations = len(arousal_labels[i])
		transformed_movie_matrix = matrix.window_matrix(movie_matrix, t, n_annotations)

		arousal_movie_matrix = matrix.sort_matrix(transformed_movie_matrix, arousal_correlations)
		valence_movie_matrix = matrix.sort_matrix(transformed_movie_matrix, valence_correlations)
		
		for arousal_model, valence_model in zip(arousal_models, valence_models):
			arousal_predictions = predict(arousal_movie_matrix, arousal_model)
			arousal_predictions = scale(arousal_predictions)

			valence_predictions = predict(valence_movie_matrix, valence_model)
			valence_predictions = scale(valence_predictions)

			if t % 2:
				a_labels = numpy.array(arousal_labels[i][t/2:-t/2], dtype = 'float')
				v_labels = numpy.array(valence_labels[i][t/2:-t/2], dtype = 'float')
				x = numpy.arange(t/2, n_annotations - t/2)
			else:
				a_labels = numpy.array(arousal_labels[i][t/2:-t/2 + 1], dtype = 'float')
				v_labels = numpy.array(valence_labels[i][t/2:-t/2 + 1], dtype = 'float')
				x = numpy.arange(t/2, n_annotations - t/2  + 1)

			a_difference = (a_labels - arousal_predictions)*(a_labels - arousal_predictions)
			v_difference = (v_labels - valence_predictions)*(v_labels - valence_predictions)
			rmse = math.sqrt(a_difference.sum()/(n_annotations - t + 1))
			coeff = numpy.corrcoef(a_labels, arousal_predictions)[0][1]
			print "Arousal rmse = %f coeff = %f" % (rmse, coeff)
			rmse = math.sqrt(v_difference.sum()/(n_annotations - t + 1))
			coeff = numpy.corrcoef(v_labels, valence_predictions)[0][1]
			print "Valence rmse = %f coeff = %f" % (rmse, coeff)
			plt.show()

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
