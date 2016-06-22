import sys
import os 
import numpy 
from sklearn.metrics import mean_squared_error
import math

def MatlabLasso(features, lables, alpha, lambdas, core = None):
   
	# write features and lables to a file
	feature_file_name = 'features_temp';
	if core is not None:
		feature_file_name += '_' + str(core)
	numpy.savetxt(feature_file_name, features, fmt='%f', delimiter=',') 
	label_file_name = 'lables_temp';
	if core is not None:
		label_file_name += '_' + str(core)
	numpy.savetxt(label_file_name, lables, fmt='%f', delimiter=',') 

	# command = 'source /usr/usc/matlab/2013a/setup.sh; /home/sabya/MATLAB/bin/matlab -nodesktop -nosplash -r "learn_ridge_regressor(' + str(alpha) + ');exit"'  
	coeffs_file_name = 'coeffs_temp'
	if core is not None:
		coeffs_file_name += '_' + str(core)
	info_file_name = 'info_temp'
	if core is not None:
		info_file_name += '_' + str(core)
	command = '/home/sabya/MATLAB/bin/matlab -nodesktop -nosplash -nodisplay -r "learn_lasso_regressor(' + str(alpha) + ',' + str(lambdas) + ',\'' + feature_file_name + '\',\'' + label_file_name  + '\',\'' + coeffs_file_name + '\',\'' + info_file_name +'\');exit"'  
	os.system(command)

	coeffs = numpy.genfromtxt(coeffs_file_name, delimiter = ',')
	fitinfo = numpy.genfromtxt(info_file_name, delimiter = ',')
	return coeffs, fitinfo

if __name__ == '__main__':
	l1 = numpy.load('../movie_matrices/After_The_Rain_valence_1.npy')
	l2 = numpy.load('../movie_matrices/Attitude_Matters_valence_1.npy')
	from load import load_output_movies
	movie_labels, _ = load_output_movies()
	labels_1 = numpy.array(movie_labels[0], dtype = 'float')
	labels_2 = numpy.array(movie_labels[1], dtype = 'float')
	print l1.shape, labels_1.shape
	coeffs, fitinfo = MatlabLasso(l1, labels_1, alpha = 0.5, lambdas = [.0001, .01, 1., 100])
	print coeffs.shape, fitinfo
	for coefficients in coeffs.T:
		print l2.shape, coefficients.shape
		predictions = numpy.dot(l2, coefficients)
		print predictions.shape
		print labels_2.shape
		print math.sqrt(mean_squared_error(predictions, labels_2))