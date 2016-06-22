import numpy as np
import matplotlib.pyplot as plt
from load import movies,load_output_movies

valence_labels_movies, arousal_labels_movies = load_output_movies()
max_valence_duration, max_arousal_duration = -10, -10
for i, movie in enumerate(movies):
	valence_labels_movies[i] = np.array(valence_labels_movies[i])
	arousal_labels_movies[i] = np.array(arousal_labels_movies[i])
	if max_valence_duration < valence_labels_movies[i].shape[0]:
		max_valence_duration = valence_labels_movies[i].shape[0]
	if max_arousal_duration < arousal_labels_movies[i].shape[0]:
		max_arousal_duration = arousal_labels_movies[i].shape[0]

for i, movie in enumerate(movies):
	x = np.arange(max_valence_duration)
	xp = np.arange(len(valence_labels_movies[i]))
	xp = xp * (max_valence_duration/len(valence_labels_movies[i]))
	valence_labels_movies[i] = np.interp(x, xp, valence_labels_movies[i])
	x = np.arange(max_arousal_duration)
	xp = np.arange(len(arousal_labels_movies[i]))
	xp = xp * (max_arousal_duration/len(arousal_labels_movies[i]))
	arousal_labels_movies[i] = np.interp(x, xp, arousal_labels_movies[i])

x = np.arange(max_valence_duration)
# for i in range(len(movies)):
plt.plot(x, valence_labels_movies[1])
plt.show()
# for i in range(len(movies)):
# 	plt.plot(x, arousal_labels_movies[i])
# 	plt.show()