import numpy as np
import math
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from load import movies, load_output_movies
import sys

def transfer_knowledge(t_array, arousal_models, valence_models):
	'''
		t_array is an array of time intervals
		arousal_models and valence_models are an array of regressors that have been trained on data1
		for every t we load the corresponding matrix of data2, where per second feature have been trained 
			on a window of t second frames
		we predict labels and scale them to -1 and 1
		Then we find the rmse and correlation with actual labels of movies
	'''
	if not len(t_array):
		return
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	for t in t_array:
		print t
		for arousal_model, valence_model in zip(arousal_models, valence_models):
			arousal_predictions_movies = np.array([], dtype = 'float')
			valence_predictions_movies = np.array([], dtype = 'float')
			arousal_labels_movies_cumulative = np.array([], dtype = 'float')
			valence_labels_movies_cumulative = np.array([], dtype = 'float')
			for i, movie in enumerate(movies):

				n_annotations = min(len(arousal_labels_movies[i]), len(valence_labels_movies[i]))
				arousal_labels = arousal_labels_movies[i][:n_annotations]
				valence_labels = valence_labels_movies[i][:n_annotations]
				if t % 2:
					arousal_labels = np.array(arousal_labels[t/2:-(t/2)], dtype = 'float')
					valence_labels = np.array(valence_labels[t/2:-(t/2)], dtype = 'float')
				else:
					arousal_labels = np.array(arousal_labels[t/2:-(t/2) + 1], dtype = 'float')
					valence_labels = np.array(valence_labels[t/2:-(t/2) + 1], dtype = 'float')

				try:
					valence_matrix = np.load('../movie_matrices/%s_valence_%d.npy' % (movie, t))
					arousal_matrix = np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, t))
				except Exception:
					print "File absent for t = %d" % t
					continue

				arousal_predictions = np.array(arousal_model.predict(arousal_matrix), dtype = 'float')
				arousal_predictions = (arousal_predictions - 1)/2 - 1
				arousal_predictions_movies = np.append(arousal_predictions_movies, arousal_predictions)
				valence_predictions = np.array(valence_model.predict(valence_matrix), dtype = 'float')
				valence_predictions = (valence_predictions - 1)/2 - 1
				valence_predictions_movies = np.append(valence_predictions_movies, valence_predictions)
				arousal_labels_movies_cumulative = np.append(arousal_labels_movies_cumulative, arousal_labels)
				valence_labels_movies_cumulative = np.append(valence_labels_movies_cumulative, valence_labels)

			# arousal_difference = arousal_predictions_movies - arousal_labels_movies_cumulative
			# arousal_product = arousal_difference * arousal_difference
			arousal_rmse = math.sqrt(mean_squared_error(arousal_predictions_movies, 
				arousal_labels_movies_cumulative))
			arousal_coeff = np.corrcoef(arousal_predictions_movies, arousal_labels_movies_cumulative)[0][1]
			# valence_difference = valence_predictions_movies - valence_labels_movies_cumulative
			# valence_product = valence_difference * valence_difference
			valence_rmse = math.sqrt(mean_squared_error(valence_predictions_movies, 
				valence_labels_movies_cumulative))
			valence_coeff = np.corrcoef(valence_predictions_movies, valence_labels_movies_cumulative)[0][1]
			print 'Arousal %25s %0.2f %0.2f' % (type(arousal_model).__name__, arousal_rmse, arousal_coeff)
			print 'Valence %25s %0.2f %0.2f' % (type(valence_model).__name__, valence_rmse, valence_coeff)
		print

if __name__ == '__main__':
	# arousal_models = [joblib.load('arousal_model_Bayesian.pkl'), joblib.load('arousal_model_ElasticNet.pkl'), 
	# 					joblib.load('arousal_model_ElasticNetCV.pkl'), joblib.load('arousal_model_DecisionTree.pkl')]
	# valence_models = [joblib.load('valence_model_Bayesian.pkl'), joblib.load('valence_model_ElasticNet.pkl'), 
	# 					joblib.load('valence_model_ElasticNetCV.pkl'), joblib.load('valence_model_DecisionTree.pkl')]
	arousal_models = [joblib.load('arousal_model_ElasticNetCV.pkl')]
	valence_models = [joblib.load('valence_model_ElasticNetCV.pkl')]
	args = sys.argv[1:]
	t_array = [int(t) for t in args]
	print t_array
	# transfer_knowledge([8, 9, 10, 11, 12, 15, 20, 30, 50, 100], arousal_models, valence_models)
	transfer_knowledge(t_array, arousal_models, valence_models)




