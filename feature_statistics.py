from load import movies
import matplotlib.pyplot as plt
import numpy as np

valence_movie_matrices, arousal_movie_matrices = [], []

for movie in movies:
	valence_movie_matrices.append(np.load('../movie_matrices/%s_valence_1.npy' % (movie)))
	arousal_movie_matrices.append(np.load('../movie_matrices/%s_arousal_1.npy' % (movie)))

for i in range(len(movies)):
	for j in range(156):
		values = valence_movie_matrices[i][:,j*9]
		plt.plot(np.arange(len(values)), values)
		plt.show()
