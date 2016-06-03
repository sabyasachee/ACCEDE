import numpy
import math
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

def correlations(arousal_values, valence_values, feature_vectors):
	# matrix = numpy.zeros((9800, 10))
	# for i in range(0, 9800):
	# 	matrix[i][0] = values[i]
	# 	for j in range(1, 10):
	# 		matrix[i][j] = feature_vectors[i][j-1]
	# result = numpy.corrcoef(matrix)
	# return list(result[0])
	mean_vectors = []
	median_vectors = []
	std_vectors = []
	kurt_vectors = []
	lower_quartile_vectors = []
	upper_quartile_vectors = []
	min_vectors = []
	max_vectors = []
	range_vectors = []
	for i in range(0, 9800):
		mean_vectors.append(feature_vectors[i][0])
		median_vectors.append(feature_vectors[i][1])
		std_vectors.append(feature_vectors[i][2])
		kurt_vectors.append(feature_vectors[i][3])
		lower_quartile_vectors.append(feature_vectors[i][4])
		upper_quartile_vectors.append(feature_vectors[i][5])
		min_vectors.append(feature_vectors[i][6])
		max_vectors.append(feature_vectors[i][7])
		range_vectors.append(feature_vectors[i][8])
	matrix = [arousal_values, valence_values, mean_vectors, median_vectors, std_vectors, kurt_vectors, lower_quartile_vectors, upper_quartile_vectors, min_vectors, 
		max_vectors, range_vectors]
	# matrix = [values, feature_vectors]
	result = numpy.corrcoef(matrix)
	return result[0:2]

def cross_validation(regressors, feature_vectors, true_values, train_to_test_ratio = 8, valid_to_test_ratio = 1):
	# return rmse, pearson-coefficient
	length = len(feature_vectors)
	dimension = len(feature_vectors[0])
	number_of_splits = train_to_test_ratio + valid_to_test_ratio + 1
	test_length = length/number_of_splits
	train_length = test_length*train_to_test_ratio
	valid_length = test_length*valid_to_test_ratio
	predict_values = []
	print "Starting cross validation"
	for i in range(0, number_of_splits):
		print "\tFold = %d" % i
		test_feature_vectors = feature_vectors[i*test_length: (i + 1)*test_length]
		test_true_values = true_values[i*test_length: (i + 1)*test_length]
		if i > 0 and i < number_of_splits - 1:
			rest_feature_vectors = feature_vectors[:i*test_length] + feature_vectors[(i+1)*test_length:]
			rest_true_values = true_values[:i*test_length] + true_values[(i+1)*test_length:]
		elif i == 0:
			rest_feature_vectors = feature_vectors[test_length:]
			rest_true_values = true_values[test_length:]
		else:
			rest_feature_vectors = feature_vectors[:-test_length]
			rest_true_values = true_values[:-test_length]
		train_feature_vectors = rest_feature_vectors[:train_length]
		valid_feature_vectors = rest_feature_vectors[train_length:]
		train_true_values = rest_true_values[:train_length]
		valid_true_values = rest_true_values[train_length:]

		tuned_regressor = None
		best_sqared_difference_sum = None
		for regressor in regressors:
			print "\t\ttuning with regressor ", type(regressor).__name__
			regressor.fit(numpy.array(train_feature_vectors, dtype = "float"), numpy.array(train_true_values, dtype = "float"))
			valid_predict_values = regressor.predict(numpy.array(valid_feature_vectors, dtype = "float"))
			squared_difference_sum = 0
			for j in range(0, valid_length):
				squared_difference_sum = (valid_predict_values[j] - valid_true_values[j])*(valid_predict_values[j] - valid_true_values[j])
			# print squared_difference_sum
			if best_sqared_difference_sum is None or best_sqared_difference_sum > squared_difference_sum:
				best_sqared_difference_sum = squared_difference_sum
				tuned_regressor = regressor
		print "\n\t\tbest regressor ", type(tuned_regressor).__name__
		# tuned_regressor.fit(numpy.array(train_feature_vectors, dtype = "float"), numpy.array(train_true_values, dtype = "float"))
		test_predict_values = tuned_regressor.predict(numpy.array(test_feature_vectors, dtype = "float"))
		predict_values += list(test_predict_values)
	print "Done"
	predict_mean = float(sum(predict_values))/length
	true_mean = float(sum(true_values))/length
	squared_difference_sum = 0
	product_sum = 0
	predict_squared_deviation_sum = 0
	true_squared_deviation_sum = 0
	for j in range(0, length):
		squared_difference_sum += (predict_values[j] - true_values[j])*(predict_values[j] - true_values[j])
		deviation_predict = predict_values[j] - predict_mean
		deviation_true = true_values[j] - true_mean
		product_sum += deviation_predict*deviation_true
		predict_squared_deviation_sum += deviation_predict*deviation_predict
		true_squared_deviation_sum += deviation_true*deviation_true
	
	rmse = math.sqrt(float(squared_difference_sum)/length)
	covariance = float(product_sum)/length
	std_predict = math.sqrt(float(predict_squared_deviation_sum)/length)
	std_true = math.sqrt(float(true_squared_deviation_sum)/length)
	correlation = float(covariance)/(std_predict*std_true)

	return rmse, correlation

def get_feature_vectors(filename):
	feature_vectors = []
	with open(filename) as fr:
		header = fr.readline()
		for i in range(0, 9800):
			line = fr.readline()
			values = line.strip().split("\t")[1:]
			features = [float(value) for value in values]
			feature_vectors.append(features)
	return feature_vectors


arousal_feature_vectors = []
valence_feature_vectors = []
valence_values = []
arousal_values = []
valence_variance_values = []
arousal_variance_values = []

with open("../features/ACCEDEfeaturesArousal_TAC2015.txt") as fr:
	header = fr.readline()
	# print header
	n_arousal_features = len(header.strip().split("\t")) - 2
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[2:]
		arousal_feature_vector = [float(value) for value in values]
		arousal_feature_vectors.append(arousal_feature_vector)
with open("../features/ACCEDEfeaturesValence_TAC2015.txt") as fr:
	header = fr.readline()
	# print header
	n_valence_features = len(header.strip().split("\t")) - 2
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[2:]
		valence_feature_vector = [float(value) for value in values]
		valence_feature_vectors.append(valence_feature_vector)
with open("../annotations/ACCEDEranking.txt") as fr:
	header = fr.readline()
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[4:]
		true_values = [float(value) for value in values]
		valence_value, arousal_value, valence_variance, arousal_variance = true_values
		valence_values.append(valence_value)
		arousal_values.append(arousal_value)
		valence_variance_values.append(valence_variance)
		arousal_variance_values.append(arousal_variance)


# Specify the regressors
regressors_1 = [LinearRegression(), 
			# ARDRegression(), 
			BayesianRidge(), 
			ElasticNet(), 
			ElasticNetCV(), 
			Lasso(), 
			KNeighborsRegressor(), 
			DecisionTreeRegressor(), 
			RandomForestRegressor(), 
			AdaBoostRegressor()
			]
regressors_2 = [SGDRegressor(loss = 'epsilon_insensitive', alpha = 0.00001),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 0.0001),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 0.001),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 0.01),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 0.1),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 1.0),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 10),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 100),
				SGDRegressor(loss = 'epsilon_insensitive', alpha = 1000)
				]

octave1_feature_vectors = get_feature_vectors("../results/octave1.txt")
octave2_feature_vectors = get_feature_vectors("../results/octave2.txt")
octave3_feature_vectors = get_feature_vectors("../results/octave3.txt")
octave4_feature_vectors = get_feature_vectors("../results/octave4.txt")
octave5_feature_vectors = get_feature_vectors("../results/octave5.txt")
octave6_feature_vectors = get_feature_vectors("../results/octave6.txt")
octave7_feature_vectors = get_feature_vectors("../results/octave7.txt")
octave8_feature_vectors = get_feature_vectors("../results/octave8.txt")
octave9_feature_vectors = get_feature_vectors("../results/octave9.txt")
octave10_feature_vectors = get_feature_vectors("../results/octave10.txt")
octave11_feature_vectors = get_feature_vectors("../results/octave11.txt")
octave12_feature_vectors = get_feature_vectors("../results/octave12.txt")

print correlations(arousal_values, valence_values, octave1_feature_vectors)
print correlations(arousal_values, valence_values, octave2_feature_vectors)
print correlations(arousal_values, valence_values, octave3_feature_vectors)
print correlations(arousal_values, valence_values, octave4_feature_vectors)
print correlations(arousal_values, valence_values, octave5_feature_vectors)
print correlations(arousal_values, valence_values, octave6_feature_vectors)
print correlations(arousal_values, valence_values, octave7_feature_vectors)
print correlations(arousal_values, valence_values, octave8_feature_vectors)
print correlations(arousal_values, valence_values, octave9_feature_vectors)
print correlations(arousal_values, valence_values, octave10_feature_vectors)
print correlations(arousal_values, valence_values, octave11_feature_vectors)
print correlations(arousal_values, valence_values, octave12_feature_vectors)