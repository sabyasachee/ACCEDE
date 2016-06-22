from load import movies, load_output_movies
import numpy as np
import matplotlib.pyplot as plt

movie_valence_labels, movie_arousal_labels = load_output_movies()
transformed_movie_valence_labels = [np.apply_along_axis(np.absolute, 0, 
	np.fft.fft(np.array(valence_labels, dtype = 'float'))) for valence_labels in movie_valence_labels]
transformed_movie_arousal_labels = [np.apply_along_axis(np.absolute, 0, 
	np.fft.fft(np.array(arousal_labels, dtype = 'float'))) for arousal_labels in movie_arousal_labels]
for i in range(0,6):
	t = np.arange(len(movie_arousal_labels[i*5]))
	plt.plot(t, movie_arousal_labels[i*5])
	plt.plot(t, transformed_movie_arousal_labels[i*5])
	plt.show()
for i in range(0,6):
	t = np.arange(len(movie_valence_labels[i*5]))
	plt.plot(t, movie_valence_labels[i*5])
	plt.plot(t, transformed_movie_valence_labels[i*5])
	plt.show()
# for i in range(0,6):
