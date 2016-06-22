import numpy as np
import os

def MatlabSVR(features, test_features, labels, core = None):
	# write features and lables to a file
	features_file_name = 'features_temp'
	if core is not None:
		features_file_name += '_' + str(core)
	np.savetxt(features_file_name, features, fmt='%f', delimiter=',') 
	test_features_file_name = 'test_features_temp'
	if core is not None:
		test_features_file_name += '_' + str(core)
	np.savetxt(test_features_file_name, test_features, fmt='%f', delimiter=',') 
	labels_file_name = 'labels_temp'
	if core is not None:
		labels_file_name += '_' + str(core)
	np.savetxt(labels_file_name, labels, fmt='%f', delimiter=',')
	test_labels_file_name = 'test_labels_temp'
	if core is not None:
		test_labels_file_name += '_' + str(core)

	command = '/home/sabya/MATLAB/bin/matlab -nodesktop -nosplash -nodisplay -r "learn_svm_regressor(\'' + features_file_name + '\',\'' + test_features_file_name + '\',\'' + labels_file_name + '\',\'' + test_labels_file_name + '\');exit;"'
	print command
	os.system(command)

	test_labels = np.genfromtxt(test_labels_file_name, delimiter = ',')
	return test_labels

if __name__ == '__main__':
	l1 = np.load('../movie_matrices/After_The_Rain_valence_1.npy')
	l2 = np.load('../movie_matrices/Attitude_Matters_valence_1.npy')
	from load import load_output_movies
	movie_labels, _ = load_output_movies()
	labels_1 = np.array(movie_labels[0], dtype = 'float')
	labels_2 = np.array(movie_labels[1], dtype = 'float')
	print MatlabSVR(l1, l2, labels_1)
	