import math
import numpy
import threading
# from mpi4py import MPI
from multiprocessing import Process

class movieloadThread(threading.Thread):
	def __init__(self, i, movie):
		threading.Thread.__init__(self)
		self.i = i
		self.movie = movie

	def run(self):
		movie_matrix = load_input_movie(self.movie)
		movies_matrix[self.i] = movie_matrix

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
				"pcm_zcr","pcm_zcr_de","pcm_zcr_stddev","pcm_zcr_amean","pcm_zcr_de_stddev","pcm_zcr_de_amean",
				"octave0",
				"octave1","octave2","octave3","octave4",
				"octave5","octave6","octave7","octave8",
				"octave9","octave10","octave11"
				]
movies = [
		"After_The_Rain", 
		"Attitude_Matters", 
		"Barely_legal_stories", 
		"Between_Viewings", 
		"Big_Buck_Bunny", 
		"Chatter", 
		"Cloudland", 
		"Damaged_Kung_Fu",
		"Decay", 
		"Elephant_s_Dream", 
		"First_Bite", 
		"Full_Service", 
		"Islands", 
		"Lesson_Learned", 
		"Norm", 
		"Nuclear_Family", 
		"On_time", 
		"Origami", 
		"Parafundit", 
		"Payload", 
		"Riding_The_Rails", 
		"Sintel", 
		"Spaceman", 
		"Superhero", 
		"Tears_of_Steel", 
		"The_room_of_franz_kafka", 
		"The_secret_number", 
		"To_Claire_From_Sonny", 
		"Wanted", 
		"You_Again"  	
		]
# movies = [
# 		"After_The_Rain"			
# 		]
movies_matrix = []
n_features = len(feature_names)
n_samples = 9800

# outputs a list of 9800 vectors each vector of length equal to number of feature_files * 9, each element of the form (feature_name, statistic, value)
# feature_name is string, statistic is a number 0-8 for mean, median,..., 
def load_input():
	print n_samples
	feature_vectors = []
	print "Loading features"
	for i in range(0, n_samples):
		feature_vectors.append([])
	for feature_name in feature_names:
		with open("../results/" + feature_name + ".txt", "r") as fr:
			header = fr.readline()
			for i in range(0, n_samples):
				line = fr.readline()
				values = line.strip().split("\t")[1:]
				feature_vector = [(feature_name,j,float(value)) for j, value in enumerate(values)]
				feature_vectors[i].extend(feature_vector)
	print "Done loading features"
	return feature_vectors

# outputs 30 matrices, one for each movie
# each matrix is of dimension n_features * n_frames
def load_input_movie(movie):
	print "Loading", movie, "..."
	movie_matrix = []
	for i, feature_name in enumerate(feature_names):
		feature_vector = []
		filename = "../movie_results/%s/%s_%s.txt" % (movie, movie, feature_name)
		with open(filename) as fr:
			line = fr.readline()
			for line in fr:
				feature_vector.append(float(line.strip().split("\t")[1]))
		movie_matrix.append(feature_vector)
	return movie_matrix

def fast_load_input_movie(movie, i):
	movie_matrix = load_input_movie(movie)
	movies_matrix[i] = movie_matrix
	print movies_matrix

def fast_load_input_movies():
	processes = []
	for i, movie in enumerate(movies):
		movies_matrix.append(None)
		processes.append(Process(target = fast_load_input_movie, args = (movie, i)))
	for process in processes:
		process.start()
	for process in processes:
		process.join()
	print 'Done loading all movies'

# outputs four lists -> valence_labels, arousal_labels, valence_correlation_coefficients and arousal_correlation_coefficients
# valence_labels and arousal_labels are lists of length 9800
# valence_correlation_coefficients and arousal_correlation_coefficients are dictionaries with key feature_name and value a list of 9 elements for each
# statistic
def load_output():
	valence_labels = []
	arousal_labels = []
	arousal_correlations = dict()
	valence_correlations = dict()

	print "Loading labels ..."
	with open("../annotations/ACCEDEranking.txt") as fr:
		line = fr.readline()
		for i in range(0, n_samples):
			values = fr.readline().strip().split("\t")[4:]
			true_values = [float(value) for value in values]
			valence_value, arousal_value, valence_variance, arousal_variance = true_values
			valence_labels.append(valence_value)
			arousal_labels.append(arousal_value)

	print "Loading valence_correlation_coefficients ..."
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

	print "Loading arousal_correlation_coefficients ..."
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

	print
	return valence_labels, arousal_labels, valence_correlations, arousal_correlations

# load valence and arousal labels for movies per second
def load_output_movies():
	valence_labels = []
	arousal_labels = []
	for movie in movies:
		labels = []
		with open("../continuous-annotations/" + movie + "_Valence.txt") as fr:
			line = fr.readline()
			for line in fr:
				value = float(line.strip().split("\t")[1])
				labels.append(value)
		valence_labels.append(labels)
		labels = []
		with open("../continuous-annotations/" + movie + "_Arousal.txt") as fr:
			line = fr.readline()
			for line in fr:
				value = float(line.strip().split("\t")[1])
				labels.append(value)
		arousal_labels.append(labels)
	return valence_labels, arousal_labels

# sort the feature_vectors according to correlation_coefficients
# and return features according to required dimension
def sort_features(feature_vectors, correlations, n_dimension = n_features * 9):

	features = []
	# sort the feature_vector according to correlations
	print "Sorting...",
	for i in range(0, n_samples):
		feature_vector = feature_vectors[i]
		feature_vector = sorted(feature_vector, key = lambda feature_tuple: correlations[feature_tuple[0]][feature_tuple[1]], reverse = True)
		feature_vectors[i] = feature_vector

	# make the features vector
	print "Filtering..."
	for i in range(0, n_samples):
		vector = [tuple_value[2] for tuple_value in feature_vectors[i][: n_dimension]]
		features.append(vector)
	return features

# load the fps of each movie file
def load_fps():
	fps = []
	with open("..//movie_results/fps.txt") as fr:
		line = fr.readline()
		for line in fr:
			value = line.strip().split("\t")[1]
			fps.append(int(value))
	return fps