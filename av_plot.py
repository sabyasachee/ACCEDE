import sys
import numpy as np
import matplotlib.pyplot as plt
from load import load_output_movies, movies

valence_labels_movies, arousal_labels_movies = load_output_movies()
valence_means, arousal_means = range(len(movies)), range(len(movies))
for i in range(len(movies)):
	valence_mean = np.mean(valence_labels_movies[i])
	arousal_mean = np.mean(arousal_labels_movies[i])
	valence_means[i] = valence_mean
	arousal_means[i] = arousal_mean
plt.plot(arousal_means, valence_means, 'bo')
plt.show()