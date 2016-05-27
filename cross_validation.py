import numpy
import math
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

def correlation(x, y):
	length = len(x)
	mean_x = float(sum(x))/length
	mean_y = float(sum(y))/length
	product_sum = 0
	squared_sum_x = 0
	squared_sum_y = 0
	for i in range(0, length):
		deviation_x = x[i] - mean_x
		deviation_y = y[i] - mean_y
		product_sum += deviation_x*deviation_y
		squared_sum_x += deviation_x*deviation_x
		squared_sum_y += deviation_y*deviation_y
	covariance = float(product_sum)/length
	std_x = math.sqrt(float(squared_sum_x)/length)
	std_y = math.sqrt(float(squared_sum_y)/length)
	try:
		covariance = float(covariance)/(std_x*std_y)
	except ZeroDivisionError as error:
		print error
	return covariance

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

# regressors = [SVR(kernel = 'linear', C = 0.00001), SVR(kernel = 'linear', C = 0.0001), SVR(kernel = 'linear', C = 0.001), SVR(kernel = 'linear', C = 0.01), 
# 			SVR(kernel = 'linear', C = 0.1), SVR(kernel = 'linear', C = 1), SVR(kernel = 'linear', C = 10), SVR(kernel = 'linear', C = 100), 
# 			SVR(kernel = 'linear', C = 1000)]
# regressors = [SVR(C = 0.00001), SVR(C = 0.0001), SVR(C = 0.001), SVR(C = 0.01), SVR(C = 0.1), SVR(C = 1), SVR(C = 10), SVR(C = 100), 
# 			SVR(C = 1000)]
# regressors = [LinearSVR(C = 0.00001), LinearSVR(C = 0.0001), LinearSVR(C = 0.001), LinearSVR(C = 0.01), 
# 			LinearSVR(C = 0.1), LinearSVR(C = 1), LinearSVR(C = 10), LinearSVR(C = 100), 
# 			LinearSVR(C = 1000)]
# regressors = [LinearSVR(C = 0.001, epsilon = 0.1), LinearSVR(C = 100, epsilon = 0.1)]
# regressors = [LinearRegression()]
# regressors = [SVR(kernel = 'linear', C = 0.001), SVR(kernel = 'linear', C = 100)]
# regressors = [SVR(kernel = 'poly', degree = 1, C = 0.001), SVR(kernel = 'poly', degree = 1, C = 100)]

audio_feature_vectors = []
arousal_feature_vectors = []
valence_feature_vectors = []
flow_feature_vectors = []
intensity_feature_vectors = []
luma_feature_vectors = []
valence_values = []
arousal_values = []
valence_variance_values = []
arousal_variance_values = []
with open("../results/audio.txt") as fr:
	header = fr.readline()
	n_audio_features = len(header.strip().split("\t")) - 1
	header = fr.readline()
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[1:]
		audio_feature_vector = [float(value) for value in values]
		audio_feature_vectors.append(audio_feature_vector)
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
with open("../results/flow.txt") as fr:
	header = fr.readline()
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[1:]
		flow_feature_vector = [float(value) for value in values]
		flow_feature_vectors.append(flow_feature_vector)
with open("../results/intensity.txt") as fr:
	header = fr.readline()
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[1:]
		intensity_feature_vector = [float(value) for value in values]
		intensity_feature_vectors.append(flow_feature_vector)
with open("../results/luma.txt") as fr:
	header = fr.readline()
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[1:]
		luma_feature_vector = [float(value) for value in values]
		luma_feature_vectors.append(flow_feature_vector)

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

# rmse, correlation = cross_validation(regressors_1, arousal_feature_vectors, arousal_values)
# print rmse, correlation

# rmse, correlation = cross_validation(regressors_1, valence_feature_vectors, valence_values)
# print rmse, correlation

# for i in range(0, n_arousal_features):
# 	features = []
# 	for j in range(0, 9800):
# 		features.append(arousal_feature_vectors[j][i])
# 	print correlation(features, arousal_values)
# print
# for i in range(0, n_valence_features):
# 	features = []
# 	for j in range(0, 9800):
# 		features.append(valence_feature_vectors[j][i])
# 	print correlation(features, valence_values)
# print
for i in range(0, 9):
	features = []
	for j in range(0, 9800):
		features.append(flow_feature_vectors[j][i])
	print numpy.corrcoef(numpy.array(features), numpy.array(valence_values))
print
for i in range(0, 9):
	features = []
	for j in range(0, 9800):
		features.append(intensity_feature_vectors[j][i])
	print numpy.corrcoef(numpy.array(features), numpy.array(valence_values))
print
for i in range(0, 9):
	features = []
	for j in range(0, 9800):
		features.append(luma_feature_vectors[j][i])
	# print correlation(features, arousal_values)
	print numpy.corrcoef(numpy.array(features), numpy.array(valence_values))

# for i in range(0, n_audio_features):
# 	for j in range(0, 9):
# 		features = []
# 		for k in range(0, 9800):
# 			features.append(audio_feature_vectors[k][i*9 + j])
# 		# print correlation(features, valence_values)
# 		print numpy.corrcoef(numpy.array(features), numpy.array(valence_values))
# 	print