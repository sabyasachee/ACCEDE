from load import movies
import numpy as np
from load import load_output_movies

def make_arff(window_length):
	with open('../movie_results/valence_movies_%d.arff' % window_length, 'w') as fw_valence:
		with open('../movie_results/arousal_movies_%d.arff' % window_length, 'w') as fw_arousal:
			fw_valence.write('@relation valence_movies_%d\n' % window_length)
			fw_arousal.write('@relation arousal_movies_%d\n' % window_length)
			
			fw_valence.write('@attribute id numeric\n')
			fw_arousal.write('@attribute id numeric\n')

			with open('../movie_results/valence_correlations_%d.txt' % window_length) as fr:
				for line in fr:
					feature, statistic, _ = line.strip().split('\t')
					fw_valence.write('@attribute %s_%s numeric\n' % (feature, statistic))
			with open('../movie_results/arousal_correlations_%d.txt' % window_length) as fr:
				for line in fr:
					feature, statistic, _ = line.strip().split('\t')
					fw_arousal.write('@attribute %s_%s numeric\n' % (feature, statistic))
			
			fw_valence.write('@attribute valence numeric\n@data\n')
			fw_arousal.write('@attribute arousal numeric\n@data\n')

			valence_labels_movies, arousal_labels_movies = load_output_movies()
			
			for i in range(len(movies)):
				n_annotations = min(len(arousal_labels_movies[i]), len(valence_labels_movies[i]))
				valence_labels_movies[i] = np.array(valence_labels_movies[i][:n_annotations], dtype = 'float')
				arousal_labels_movies[i] = np.array(arousal_labels_movies[i][:n_annotations], dtype = 'float')
				
				if window_length > 2:
					if window_length % 2:
						valence_labels_movies[i] = valence_labels_movies[i][window_length/2:-(window_length/2)]
						arousal_labels_movies[i] = arousal_labels_movies[i][window_length/2:-(window_length/2)]
					else:
						valence_labels_movies[i] = valence_labels_movies[i][window_length/2:-(window_length/2) + 1]
						arousal_labels_movies[i] = arousal_labels_movies[i][window_length/2:-(window_length/2) + 1]
				elif window_length == 2:
					valence_labels_movies[i] = valence_labels_movies[i][1:]
					arousal_labels_movies[i] = arousal_labels_movies[i][1:]

			for i, movie in enumerate(movies):
				print "Writing movie", movie
				valence_matrix = np.load('../movie_matrices_movie_features/%s_valence_%d.npy' % (movie, window_length))
				arousal_matrix = np.load('../movie_matrices_movie_features/%s_arousal_%d.npy' % (movie, window_length))
				valence_labels = valence_labels_movies[i]
				arousal_labels = arousal_labels_movies[i]
				n_samples = valence_labels.shape[0]
				for j in range(n_samples):
					valence_features = ','.join(map(str, valence_matrix[j]))
					arousal_features = ','.join(map(str, arousal_matrix[j]))
					valence_line = str(i) + ',' + valence_features + ',' + str(valence_labels[j]) + '\n'
					arousal_line = str(i) + ',' + arousal_features + ',' + str(arousal_labels[j]) + '\n'
					fw_valence.write(valence_line)
					fw_arousal.write(arousal_line)

if __name__ == '__main__':
	make_arff(1)
	make_arff(2)
	make_arff(3)
	make_arff(4)
	make_arff(5)