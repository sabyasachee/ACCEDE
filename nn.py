from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, GaussianLayer, SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np
import math
from load import movies, load_output_movies
from sklearn.metrics import mean_squared_error as MSE
from multiprocessing import Process, Array
import warnings
import sys

def fold_training(start_i, end_i, n_train, n_valid, predictions, matrices, labels_list):

	hidden_sizes = [1,2]
	n_neurons_list = [100, 200, 300, 400, 500, 600, 700]
	layer_classes = [SigmoidLayer, TanhLayer]
	epochs = 10
	n_features = matrices[0].shape[1]
	delta_rmse = 0.03

	for i in range(start_i, end_i):
		sys.stdout = open('NN_fold_%d.txt' % i, 'w')
		print "Fold %d > starting" % i
		train_dataset = SupervisedDataSet(n_features, 1)
		valid_dataset = SupervisedDataSet(n_features, 1)
		test_dataset = SupervisedDataSet(n_features, 1)

		rest_matrices = matrices[:i] + matrices[i + 1:]
		rest_labels_list = labels_list[:i] + labels_list[i + 1:]
		train_matrices = rest_matrices[:n_train]
		train_labels_list = rest_labels_list[:n_train]
		valid_matrices = rest_matrices[n_train:]
		valid_labels_list = rest_labels_list[n_train:]
		test_matrix = matrices[i]
		test_labels = labels_list[i]
		train_matrix = np.vstack(tuple(train_matrices))
		train_labels = np.hstack(tuple(train_labels_list))
		valid_matrix = np.vstack(tuple(valid_matrices))
		valid_labels = np.hstack(tuple(valid_labels_list))
		print 'Fold %d > done concatenation' % i

		train_dataset.setField('input', train_matrix)
		train_dataset.setField('target', train_labels.reshape(-1,1))
		valid_dataset.setField('input', valid_matrix)
		valid_dataset.setField('target', np.zeros((valid_labels.shape[0],1)))
		test_dataset.setField('input', test_matrix)
		test_dataset.setField('target', np.zeros((test_labels.shape[0],1)))
		print 'Fold %d > datasets prepared' % i

		offset = 0
		labels_sublist = labels_list[:i]
		for list_item in labels_sublist:
			offset += list_item.shape[0]	

		min_rmse = None
		best_correlation = None
		best_n_neurons = None
		best_layer_class = None
		best_net = None
		for layer_class in layer_classes:
			for n_neurons in n_neurons_list:
				net = buildNetwork(n_features, n_neurons, 1, bias = True, hiddenclass = layer_class)
				trainer = BackpropTrainer(net, train_dataset, weightdecay = 0.01, learningrate = 0.005)
				print '\t\tFold %d > training n_neurons %d layer_class %s' % (i, n_neurons, layer_class.__name__)
				with warnings.catch_warnings():
					warnings.filterwarnings('error')
					try:
						trainer.trainEpochs(epochs = epochs)
						# trainer.trainUntilConvergence()
					except Warning as e:
						print '\t\tFold %d > Runtime Warning encountered' % i
						continue
				valid_predictions = net.activateOnDataset(valid_dataset)
				valid_predictions = valid_predictions.flatten()
				rmse = math.sqrt(MSE(valid_predictions, valid_labels))
				corr = np.corrcoef(valid_predictions, valid_labels)[0][1]
				print '\t\tFold %d > valid set rmse %f correlation %f' % (i, rmse, corr)
				if min_rmse is None or (rmse < min_rmse - delta_rmse) or (rmse > min_rmse - delta_rmse and 
					rmse < min_rmse + delta_rmse and corr > best_correlation ):
					min_rmse = rmse
					best_correlation = corr
					best_n_neurons = n_neurons
					best_layer_class = layer_class
					best_net = net

		print 'Fold %d > best n_neurons %d best layer_class %s' % (i, best_n_neurons, best_layer_class.__name__)
		test_predictions = np.array(best_net.activateOnDataset(test_dataset))
		test_predictions = test_predictions.flatten()
		rmse = math.sqrt(MSE(test_predictions, test_labels))
		corr = np.corrcoef(test_predictions, test_labels)[0][1]
		print 'Fold %d > test set rmse %f correlation %f' % (i, rmse, corr)
		np.save('NN_fold_%d.npy' % i, test_predictions)

		for j, prediction in enumerate(test_predictions):
			predictions[offset + j] = prediction
		print 'Fold %d > done' % i

if __name__ == '__main__':
	# sys.stdout = open('output8.txt', 'w')
	valence_t = 3
	arousal_t = 1
	threshold = 0.18
	valence_labels_movies, arousal_labels_movies = load_output_movies()
	valence_movie_matrices, arousal_movie_matrices = [], []

	for movie in movies:
		valence_movie_matrices.append(np.load('../movie_matrices/%s_valence_%d.npy' % (movie, 
			valence_t)))
		arousal_movie_matrices.append(np.load('../movie_matrices/%s_arousal_%d.npy' % (movie, 
			arousal_t)))

	for i in range(len(movies)):
		n_annotations = min(len(valence_labels_movies[i]), len(arousal_labels_movies[i]))
		valence_labels_movies[i] = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')
		arousal_labels_movies[i] = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')

	for i in range(len(movies)):
		if valence_t > 2:
			if valence_t % 2:
				valence_labels_movies[i] = valence_labels_movies[i][valence_t/2:-(valence_t/2)]
			else:
				valence_labels_movies[i] = valence_labels_movies[i][valence_t/2:-(valence_t/2) + 1]
		elif valence_t == 2:
			valence_labels_movies[i] = valence_labels_movies[i][1:]

		if arousal_t > 2:
			if arousal_t % 2:
				arousal_labels_movies[i] = arousal_labels_movies[i][arousal_t/2:-(arousal_t/2)]
			else:
				arousal_labels_movies[i] = arousal_labels_movies[i][arousal_t/2:-(arousal_t/2) + 1]
		elif arousal_t == 2:
			arousal_labels_movies[i] = arousal_labels_movies[i][1:]

	n_folds = 30
	n_cores = 5
	processes = []
	valence_labels = np.hstack(tuple(valence_labels_movies))
	valence_predictions = Array('d', len(valence_labels))
	for i in range(0, n_cores):
		process = Process(target = fold_training, args = ((n_folds*i)/n_cores, (n_folds*(i + 1))/n_cores, 
			25, 4, valence_predictions, valence_movie_matrices, valence_labels_movies))
		processes.append(process)
		print 'process %d started' % (i)
		process.start()

	for j, process in enumerate(processes):
		process.join()
		print 'process', j, 'joined'
	print "All processes joined"

	valence_predictions = np.array(valence_predictions, dtype = 'float')
	print 'Final valence rmse %f correlation %f' % (math.sqrt(MSE(valence_predictions, valence_labels)), 
			np.corrcoef(valence_predictions, valence_labels)[0][1])
		