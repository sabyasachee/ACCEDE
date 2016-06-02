import math
import numpy
from sklearn.linear_model import LinearRegression
from EvalModel import TrainModel, TestModel, WekaSVR

MEAN = 0
MEDIAN = 1
STD = 2
KURT = 3
LOW_QUART = 4
UP_QUART = 5
MIN = 6
MAX = 7
RANGE = 8

def cross_validation(c_vals, feature_vectors, labels, correlations, k = 1):

	n_dimension = k*n_features
	features = []
	# sort the feature_vector according to correlations
	for i in range(0, n_samples):
		feature_vector = feature_vectors[i]
		feature_vector = sorted(feature_vector, key = lambda feature_tuple: correlations[feature_tuple[0]][feature_tuple[1]], reverse = True)
		feature_vectors[i] = feature_vector

	# make the features vector
	for i in range(0, n_samples):
		vector = [tuple_value[2] for tuple_value in feature_vectors[i][:n_dimension]]
		features.append(vector)

	length = len(features)
	number_of_splits = 10
	test_length = length/number_of_splits
	train_length = test_length
	valid_length = test_length*8
	predict_labels = []

	# Iterating over each split
	print "Starting cross validation"
	for i in range(0, number_of_splits):
		print "\tFold = %d" % i
		# calculate the test features, train_features and valid_features
		test_features = features[i*test_length: (i + 1)*test_length]
		test_labels = labels[i*test_length: (i + 1)*test_length]
		if i > 0 and i < number_of_splits - 1:
			rest_features = features[:i*test_length] + features[(i+1)*test_length:]
			rest_labels = labels[:i*test_length] + labels[(i+1)*test_length:]
		elif i == 0:
			rest_features = features[test_length:]
			rest_labels = labels[test_length:]
		else:
			rest_features = features[:-test_length]
			rest_labels = labels[:-test_length]
		train_features = rest_features[:train_length]
		valid_features = rest_features[train_length:]
		train_labels = rest_labels[:train_length]
		valid_labels = rest_labels[train_length:]

		min_squared_difference_sum = None
		best_c_val = None
		# try with every c_val and save the best_c_val if the squared difference sum (mse) is least
		for c_val in c_vals:
			print "\t\t c_val = ", c_val
			model = WekaSVR()
			model.fit(numpy.array(train_features, dtype = "float"), numpy.array(train_labels, dtype = "float"), c_val = c_val)
			valid_predict_labels_string = model.predict(numpy.array(valid_features, dtype = "float"))
			valid_predict_labels = [float(x) for x in valid_predict_labels_string]
			squared_difference_sum = 0
			for j in range(0, valid_length):
				squared_difference_sum = (valid_predict_labels[j] - valid_labels[j])*(valid_predict_labels[j] - valid_labels[j])
			# print squared_difference_sum
			if min_squared_difference_sum is None or min_squared_difference_sum > squared_difference_sum:
				min_squared_difference_sum = squared_difference_sum
				best_c_val = c_val

		print "\n\t\tbest c_val ", best_c_val
		best_model = WekaSVR(remove_model_file = False, force_model_filename = 'WekaSVRFold' + str(i))
		best_model.fit(numpy.array(train_features, dtype = "float"), numpy.array(train_labels, dtype = "float"), c_val = best_c_val)
		test_predict_labels_string = best_model.predict(numpy.array(test_features, dtype = "float"))
		test_predict_labels = [float(x) for x in test_predict_labels_string]
		predict_labels += list(test_predict_labels)
	
	print "Done"
	predict_mean = float(sum(predict_labels))/length
	true_mean = float(sum(labels))/length
	squared_difference_sum = 0
	product_sum = 0
	predict_squared_deviation_sum = 0
	true_squared_deviation_sum = 0
	for j in range(0, length):
		squared_difference_sum += (predict_labels[j] - labels[j])*(predict_labels[j] - labels[j])
		deviation_predict = predict_labels[j] - predict_mean
		deviation_true = labels[j] - true_mean
		product_sum += deviation_predict*deviation_true
		predict_squared_deviation_sum += deviation_predict*deviation_predict
		true_squared_deviation_sum += deviation_true*deviation_true
	
	rmse = math.sqrt(float(squared_difference_sum)/length)
	covariance = float(product_sum)/length
	std_predict = math.sqrt(float(predict_squared_deviation_sum)/length)
	std_true = math.sqrt(float(true_squared_deviation_sum)/length)
	correlation = float(covariance)/(std_predict*std_true)

	return rmse, correlation

# all the features
feature_names = [
				"luma", "intensity", "flow", 
				"mfcc[1]","mfcc_de[1]","mfcc_de_de[1]","mfcc[1]_stddev","mfcc[1]_amean","mfcc_de[1]_stddev","mfcc_de[1]_amean","mfcc_de_de[1]_stddev","mfcc_de_de[1]_amean",
				"mfcc[2]","mfcc_de[2]","mfcc_de_de[2]","mfcc[2]_stddev","mfcc[2]_amean","mfcc_de[2]_stddev","mfcc_de[2]_amean","mfcc_de_de[2]_stddev","mfcc_de_de[2]_amean",
				"mfcc[3]","mfcc_de[3]","mfcc_de_de[3]","mfcc[3]_stddev","mfcc[3]_amean","mfcc_de[3]_stddev","mfcc_de[3]_amean","mfcc_de_de[3]_stddev","mfcc_de_de[3]_amean",
				"mfcc[4]","mfcc_de[4]","mfcc_de_de[4]","mfcc[4]_stddev","mfcc[4]_amean","mfcc_de[4]_stddev","mfcc_de[4]_amean","mfcc_de_de[4]_stddev","mfcc_de_de[4]_amean",
				"mfcc[5]","mfcc_de[5]","mfcc_de_de[5]","mfcc[5]_stddev","mfcc[5]_amean","mfcc_de[5]_stddev","mfcc_de[5]_amean","mfcc_de_de[5]_stddev","mfcc_de_de[5]_amean",
				"mfcc[6]","mfcc_de[6]","mfcc_de_de[6]","mfcc[6]_stddev","mfcc[6]_amean","mfcc_de[6]_stddev","mfcc_de[6]_amean","mfcc_de_de[6]_stddev","mfcc_de_de[6]_amean",
				"mfcc[7]","mfcc_de[7]","mfcc_de_de[7]","mfcc[7]_stddev","mfcc[7]_amean","mfcc_de[7]_stddev","mfcc_de[7]_amean","mfcc_de_de[7]_stddev","mfcc_de_de[7]_amean",
				"mfcc[8]","mfcc_de[8]","mfcc_de_de[8]","mfcc[8]_stddev","mfcc[8]_amean","mfcc_de[8]_stddev","mfcc_de[8]_amean","mfcc_de_de[8]_stddev","mfcc_de_de[8]_amean",
				"mfcc[9]","mfcc_de[9]","mfcc_de_de[9]","mfcc[9]_stddev","mfcc[9]_amean","mfcc_de[9]_stddev","mfcc_de[9]_amean","mfcc_de_de[9]_stddev","mfcc_de_de[9]_amean",
				"mfcc[10]","mfcc_de[10]","mfcc_de_de[10]","mfcc[10]_stddev","mfcc[10]_amean","mfcc_de[10]_stddev","mfcc_de[10]_amean","mfcc_de_de[10]_stddev",
					"mfcc_de_de[10]_amean",
				"mfcc[11]","mfcc_de[11]","mfcc_de_de[11]","mfcc[11]_stddev","mfcc[11]_amean","mfcc_de[11]_stddev","mfcc_de[11]_amean","mfcc_de_de[11]_stddev",
					"mfcc_de_de[11]_amean",
				"mfcc[12]","mfcc_de[12]","mfcc_de_de[12]","mfcc[12]_stddev","mfcc[12]_amean","mfcc_de[12]_stddev","mfcc_de[12]_amean","mfcc_de_de[12]_stddev",
					"mfcc_de_de[12]_amean",
				"pcm_LOGenergy","pcm_LOGenergy_de","pcm_LOGenergy_de_de","pcm_LOGenergy_stddev","pcm_LOGenergy_amean","pcm_LOGenergy_de_stddev","pcm_LOGenergy_de_amean",
					"pcm_LOGenergy_de_de_stddev","pcm_LOGenergy_de_de_amean",
				"voiceProb","voiceProb_de","voiceProb_stddev","voiceProb_amean","voiceProb_de_stddev","voiceProb_de_amean",
				"HNR","HNR_de","HNR_stddev","HNR_amean","HNR_de_stddev","HNR_de_amean",
				"F0", "F0_de","F0_stddev","F0_amean","F0_de_stddev","F0_de_amean",
				"pcm_zcr","pcm_zcr_de","pcm_zcr_stddev","pcm_zcr_amean","pcm_zcr_de_stddev","pcm_zcr_de_amean"
				]
n_features = len(feature_names)
n_samples = 9800
# dictionary of correlation coefficients, key --> [mean, median, std, kurtosis, lower_quartile, upper_quartile, min, max, range]
arousal_correlations = dict()
valence_correlations = dict()
feature_vectors = []
feature_header = []
valence_labels = []
arousal_labels = []

# load valence_correlations and arousal_correlations
with open("../results/valence_correlations.txt") as fr:
	line = fr.readline()
	for i in range(0, n_features):
		line = fr.readline()
		values = line.strip().split("\t")
		feature_name = values[0]
		correlations = [float(value) for value in values[1:]]
		for j, value in enumerate(correlations):
			if math.isnan(value):
				correlations[j] = -10
		valence_correlations[feature_name] = correlations

with open("../results/arousal_correlations.txt") as fr:
	line = fr.readline()
	for i in range(0, n_features):
		line = fr.readline()
		values = line.strip().split("\t")
		feature_name = values[0]
		correlations = [float(value) for value in values[1:]]
		for j, value in enumerate(correlations):
			if math.isnan(value):
				correlations[j] = -10
		arousal_correlations[feature_name] = correlations

# load valence and aruosal labels
with open("../annotations/ACCEDEranking.txt") as fr:
	line = fr.readline()
	for i in range(0, n_samples):
		values = fr.readline().strip().split("\t")[4:]
		true_values = [float(value) for value in values]
		valence_value, arousal_value, valence_variance, arousal_variance = true_values
		valence_labels.append(valence_value)
		arousal_labels.append(arousal_value)

# feature_vectors shape = n_samples * (n_features * 9), columns are ordered in decreasing order of correlation coefficient
# Each element is a tuple (feature_name, statistic_number, value)
# feature_header shape = (n_features * 9), each a tuple (feature_name (string), number (0-8) telling statistics) ordered --do--
for i in range(0, n_samples):
	feature_vectors.append([])
for feature_name in feature_names:
	for i in range(0, 9):
		feature_header.append((feature_name, i))
	with open("../results/" + feature_name + ".txt", "r") as fr:
		header = fr.readline()
		for i in range(0, n_samples):
			line = fr.readline()
			values = line.strip().split("\t")[1:]
			feature_vector = [(feature_name,j,float(value)) for j, value in enumerate(values)]
			feature_vectors[i].extend(feature_vector)

# feature_header = sorted(feature_header, key = lambda feature_tuple: valence_correlations[feature_tuple[0]][feature_tuple[1]], reverse = True)




# model = TrainModel(train_features, train_labels, 'WekaSVR', [0.001, 0.1])
# print "testing model"
# model_file = 'temp_model15:17:53.1239460.690601809624'
# model = WekaSVR(model_file)
# print TestModel(model, test_features)
# print test_labels

c_vals = [0.00001 ,0.0001, 0.001, 0.01, 0.1, 1., 10.]

print cross_validation(c_vals, feature_vectors, valence_labels, valence_correlations, k = 1)