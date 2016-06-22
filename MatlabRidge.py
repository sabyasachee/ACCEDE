import sys
import os 
import numpy 

def MatlabRidge(features, lables, alpha, core = None):
   
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
	command = '/home/sabya/MATLAB/bin/matlab -nodesktop -nosplash -nodisplay -r "learn_ridge_regressor(' + str(alpha) + ',\'' + feature_file_name + '\',\'' + label_file_name  + '\',\'' + coeffs_file_name + '\');exit"'  
	os.system(command)

	coeffs = numpy.genfromtxt(coeffs_file_name, delimiter = ',')
	return coeffs

if __name__ == '__main__':
	l = numpy.load('../movie_matrices/After_The_Rain_valence_1.npy')
	from load import load_output_movies
	movie_labels, _ = load_output_movies()
	labels = numpy.array(movie_labels[0], dtype = 'float')
	print l.shape, labels.shape
	coeffs = MatlabRidge(l, labels, alpha = 1000)
	predictions = coeffs*l
	print predictions.shape