import csv
import numpy as np
from load import movies, load_output_movies
import os

movie_valence_labels, movie_arousal_labels = load_output_movies()
filenames = sorted(os.listdir('../continuous-physiological/'))
for i,movie in enumerate(filenames):
	if movie.endswith('.csv'):
		filename = '../continuous-physiological/' + movie
		valence_labels = movie_valence_labels[i]
		arousal_labels = movie_arousal_labels[i]
		with open(filename) as fr:
			reader = csv.reader(fr)
			gsr_values = []
			arousal_values = []
			for row in reader:
				second, gsr, arousal = row[0].split(';')
				second = int(second.strip())
				gsr = float(gsr.strip())
				arousal = float(arousal.strip())
				gsr_values.append(gsr)
				arousal_values.append(arousal_labels[second - 1])
			arousal_values = np.array(arousal_values, dtype = 'float')
			gsr_values = np.array(gsr_values, dtype = 'float')
			print np.corrcoef(arousal_values, gsr_values)[0][1]