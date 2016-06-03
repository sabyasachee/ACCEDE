import numpy
from sklearn.externals import joblib
import load
import utils

def transfer_knowledge(valence_model, arousal_model, t = 10):
	# load labels and movies
	valence_labels, arousal_labels = load.load_output_movies()
	videos_fps = load.load_fps()
	movies = load.movies
	load.fast_load_input_movies()
	movies_matrix = load.movies_matrix
	
	# load valence_header and arousal_headers
	feature_names = load.feature_names
	feature_name_to_index = {}
	valence_header = []
	arousal_header = []
	for i, feature_name in enumerate(feature_names):
		feature_name_to_index[feature_name] = i
		for j in range(0, 9):
			valence_header.append((feature_name, j))
			arousal_header.append((feature_name, j))
	_, _, valence_correlations, arousal_correlations = load.load_output()

	# sort valence_header and arousal_header
	valence_header = sorted(valence_header, key = lambda valence_tuple: valence_correlations[valence_tuple[0]][valence_tuple[1]], reverse = True)
	arousal_header = sorted(arousal_header, key = lambda arousal_tuple: arousal_correlations[arousal_tuple[0]][arousal_tuple[1]], reverse = True)

	# iterate through each movie
	# get the labels
	# for each second, construct a vector of length n_features*9 according to header
	# predict using model
	# collect all predictions
	# scale predictions
	# calculate rmse and correlation
	for i, movie_matrix in movies_matrix:
		valence_labels_movie = valence_labels[i]
		arousal_labels_movie = arousal_labels[i]
		video_fps = videos_fps[i]
		audio_fps = 100
		length = len(valence_labels_movie)

		valence_predict_movie = []
		arousal_predict_movie = []

		for j in range(0, length):
			features = []
			for (name, statistic) in valence_header:
				k = feature_name_to_index[name]
				if name in ["luma", "intensity", "flow"]:
					fps = video_fps
				else:
					fps = audio_fps
				last_frame = len(movies_matrix[k])
				start_frame = int(round(j*fps))
				end_frame = int(round((j + 1)*fps))
				if end_frame > last_frame:
					end_frame = last_frame
				values = movies_matrix[k][start_frame: end_frame]
				value = utils.statistic(values, statistic)
				features.append(values)
			predict = valence_model.predict(features)
			valence_predict_movie.append(predict)

			features = []
			for (name, statistic) in arousal_header:
				k = feature_name_to_index[name]
				if name in ["luma", "intensity", "flow"]:
					fps = video_fps
				else:
					fps = audio_fps
				last_frame = len(movies_matrix[k])
				start_frame = int(round(j*fps))
				end_frame = int(round((j + 1)*fps))
				if end_frame > last_frame:
					end_frame = last_frame
				values = movies_matrix[k][start_frame: end_frame]
				value = utils.statistic(values, statistic)
				features.append(values)
			predict = arousal_model.predict(features)
			arousal_predict_movie.append(predict)

		valence_predict_movie = utils.scale(valence_predict_movie)
		arousal_predict_movie = utils.scale(arousal_predict_movie)


transfer_knowledge()
