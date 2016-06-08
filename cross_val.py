import math
import numpy

def simple_cv(features, labels, regressors):
	length = len(features)
	dimension = len(features[0])
	number_of_splits = 10
	test_length = length/number_of_splits
	train_length = test_length*8
	valid_length = test_length
	predict_labels = []
	print "Starting cross validation"
	for i in range(0, number_of_splits):
		print "\tFold = %d" % i
		test_features = features[i*test_length: (i + 1)*test_length]
		test_labels = labels[i*test_length: (i + 1)*test_length]
		rest_features = features[:i*test_length] + features[(i+1)*test_length:]
		rest_labels = labels[:i*test_length] + labels[(i+1)*test_length:]
		train_features = rest_features[:train_length]
		valid_features = rest_features[train_length:]
		train_labels = rest_labels[:train_length]
		valid_labels = rest_labels[train_length:]

		best_regressor = None
		best_sqared_difference_sum = None
		for regressor in regressors:
			print "\t\tRegressor ", type(regressor).__name__,
			# print train_features
			regressor.fit(numpy.array(train_features, dtype = "float"), numpy.array(train_labels, dtype = "float"))
			valid_predict_labels = regressor.predict(numpy.array(valid_features, dtype = "float"))
			squared_difference_sum = 0
			for j in range(0, valid_length):
				squared_difference_sum += (valid_predict_labels[j] - valid_labels[j])*(valid_predict_labels[j] - valid_labels[j])
			print "rmse ", math.sqrt(float(squared_difference_sum)/valid_length)
			if best_sqared_difference_sum is None or best_sqared_difference_sum > squared_difference_sum:
				best_sqared_difference_sum = squared_difference_sum
				best_regressor = regressor
		print "\t\tbest regressor ", best_regressor, "\n"
		# best_regressor.fit(numpy.array(train_features, dtype = "float"), numpy.array(train_labels, dtype = "float"))
		test_predict_labels = best_regressor.predict(numpy.array(test_features, dtype = "float"))
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