import numpy as np
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.externals import joblib
from load import movies, load_output_movies
from matrix import join_matrices, join_vectors, rmse_matrix, extrapolate
from multiprocessing import Process, Queue

def baseline_prediction(valence_t, arousal_t, valence_model, arousal_model, valence_labels_movies, 
	arousal_labels_movies):
	valence_movie_matrices = []
	arousal_movie_matrices = []
	diff_valence_labels_movies = []
	diff_arousal_labels_movies = []
	for i, movie in enumerate(movies):
		valence_matrix 		= np.load('../movie_matrices/%s_valence_%d.npy' % (movie, valence_t))
		arousal_matrix 		= np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, arousal_t))
		valence_labels 		= valence_labels_movies[i]
		valence_predictions = valence_model.predict(valence_matrix)
		arousal_labels 		= arousal_labels_movies[i]
		arousal_predictions = arousal_model.predict(arousal_matrix)
		valence_predictions = extrapolate(len(valence_labels), valence_predictions)
		arousal_predictions = extrapolate(len(arousal_labels), arousal_predictions)
		diff_valence_labels = valence_labels - valence_predictions
		diff_arousal_labels = arousal_labels - arousal_predictions
		diff_valence_labels_movies.append(diff_valence_labels)
		diff_arousal_labels_movies.append(diff_arousal_labels)
	return diff_valence_labels_movies, diff_arousal_labels_movies

# def shift_training():


def fold_training(valence_queue, arousal_queue, i, valence_regressors, arousal_regressors, valence_movie_matrices, 
	arousal_movie_matrices, valence_labels_movies, arousal_labels_movies, 
	n_test_matrices, n_train_matrices, n_valid_matrices):

	print "Fold %d" % i
	# queue.put((i, "valence", i))
	# queue.put((i, "arousal", 2*i))
	# return
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

	print "Fold %d length of valence_labels %d, arousal_labels %d" % (i, len(valence_test_labels), 
		len(arousal_test_labels))

	valence_rmse = None
	arousal_rmse = None
	valence_model = None
	arousal_model = None
	for regressor in valence_regressors:
		print "Fold %d Valence regressor %s" % (i, type(regressor).__name__)
		regressor.fit(valence_train_matrix, valence_train_labels)
		valence_valid_predictions = regressor.predict(valence_valid_matrix)
		rmse = rmse_matrix(valence_valid_predictions, valence_valid_labels)
		if valence_rmse is None or valence_rmse > rmse:
			valence_rmse = rmse
			valence_model = regressor

	for regressor in arousal_regressors:
		print "Fold %d Arousal regressor %s" % (i, type(regressor).__name__)
		regressor.fit(arousal_train_matrix, arousal_train_labels)
		arousal_valid_predictions = regressor.predict(arousal_valid_matrix)
		rmse = rmse_matrix(arousal_valid_predictions, arousal_valid_labels)
		if arousal_rmse is None or arousal_rmse > rmse:
			arousal_rmse = rmse
			arousal_model = regressor

	print "Fold %d Finding predictions" % i
	# valence_model.fit(valence_train_matrix, valence_train_labels)
	valence_test_predictions = valence_model.predict(valence_test_matrix)
	# valence_predictions = np.append(valence_predictions, valence_test_predictions)
	# arousal_model.fit(arousal_train_matrix, arousal_train_labels)
	arousal_test_predictions = arousal_model.predict(arousal_test_matrix)
	# arousal_predictions = np.append(arousal_predictions, arousal_test_predictions)
	# queue.put(valence_test_predictions)
	# print queue.qsize()
	# queue.put(arousal_test_predictions)
	# print queue.qsize()
	for valence_test_prediction in valence_test_predictions:
		pass
		# valence_queue.put(valence_test_prediction)
	for arousal_test_prediction in arousal_test_predictions:
		pass
		# arousal_queue.put(arousal_test_prediction)
	print valence_queue.qsize(), arousal_queue.qsize()
	print "Fold %d Done" % i
	return

def simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies):
	n_train_matrices = 21
	n_valid_matrices = 6
	n_test_matrices = 3
	# valence_predictions_vectors = []
	# arousal_predictions_vectors = []
	valence_predictions = np.array([], dtype = 'float')
	arousal_predictions = np.array([], dtype = 'float')
	valence_labels = join_vectors(valence_labels_movies)
	arousal_labels = join_vectors(arousal_labels_movies)
	valence_queues = []
	arousal_queues = []
	processes = []
	for i in range(0, 10):
		# valence_predictions_vectors.append([])
		# arousal_predictions_vectors.append([])
		valence_queue = Queue()
		arousal_queue = Queue()
		valence_queues.append(valence_queue)
		arousal_queues.append(arousal_queue)
		process = Process(target = fold_training, args = (valence_queue, arousal_queue, i, 
			valence_regressors, arousal_regressors, 
			valence_movie_matrices, arousal_movie_matrices, 
			valence_labels_movies, arousal_labels_movies, 
			n_test_matrices, n_train_matrices, n_valid_matrices,))
		processes.append(process)
		process.start()

	for process in processes:
		process.join()
	print "All processes joined"

	for valence_queue, arousal_queue in zip(valence_queues, arousal_queues):
		while not valence_queue.empty():
			valence_predictions = np.append(valence_predictions, valence_queue.get())
		while not arousal_queue.empty():
			arousal_predictions = np.append(arousal_predictions, arousal_queue.get())
		# i, name, vector = queue.get()
		# if name == 'arousal':
		# 	arousal_predictions_vectors[i] = vector
		# else:
		# 	valence_predictions_vectors[i] = vector
		# valence_predictions_vectors.append(queues[i].get())
		# arousal_predictions_vectors.append(queues[i].get())

	# valence_predictions = join_vectors(valence_predictions_vectors)
	# arousal_predictions = join_vectors(arousal_predictions_vectors)

	print rmse_matrix(valence_labels, valence_predictions), np.corrcoef(valence_labels, 
		valence_predictions)[0][1]
	print rmse_matrix(arousal_labels, arousal_predictions), np.corrcoef(arousal_labels, 
		arousal_predictions)[0][1]

# regressors = [LinearRegression(), ElasticNetCV(max_iter = 10000), BayesianRidge(), ElasticNet(), 
# 	DecisionTreeRegressor(max_depth = 2)]
# valence_regressors = [ElasticNetCV(max_iter = 10000)]
# arousal_regressors = [ElasticNetCV(max_iter = 10000)]
valence_regressors = [LinearRegression()]
arousal_regressors = [LinearRegression()]
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

diff_valence_labels_movies, diff_arousal_labels_movies = baseline_prediction(valence_t, arousal_t, 
	valence_model, arousal_model, valence_labels_movies, arousal_labels_movies)
print "Baseline Prediction Done"
simple_cv(valence_regressors, arousal_regressors, valence_movie_matrices, arousal_movie_matrices, 
	valence_labels_movies, arousal_labels_movies)

