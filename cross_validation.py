import numpy
import math
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

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
			print "\t\ttuning with regressor C = %f\n" % regressor.C
			regressor.fit(numpy.array(train_feature_vectors), numpy.array(train_true_values))
			valid_predict_values = regressor.predict(numpy.array(valid_feature_vectors))
			squared_difference_sum = 0
			for j in range(0, valid_length):
				squared_difference_sum = (valid_predict_values[j] - valid_true_values[j])*(valid_predict_values[j] - valid_true_values[j])
			if best_sqared_difference_sum is None or best_sqared_difference_sum > squared_difference_sum:
				best_sqared_difference_sum = squared_difference_sum
				tuned_regressor = regressor

		test_predict_values = tuned_regressor.predict(numpy.array(test_feature_vectors))
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
regressors = [LinearSVR(C = 0.00001), LinearSVR(C = 0.0001), LinearSVR(C = 0.001), LinearSVR(C = 0.01), 
			LinearSVR(C = 0.1), LinearSVR(C = 1), LinearSVR(C = 10), LinearSVR(C = 100), 
			LinearSVR(C = 1000)]
# regressors = [LinearRegression()]
feature_vectors = []
valence_values = []
arousal_values = []
valence_variance_values = []
arousal_variance_values = []
with open("../features/ACCEDEfeaturesArousal_TAC2015.txt") as fr:
	header = fr.readline()
	for i in range(0, 9800):
		values = fr.readline().strip().split("\t")[2:]
		feature_vector = [float(value) for value in values]
		feature_vectors.append(feature_vector)
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
# print valence_values
# print feature_vectors
# print feature_vectors[0]
# print feature_vectors[-1]
rmse, correlation = cross_validation(regressors, feature_vectors, valence_values)
print rmse, correlation