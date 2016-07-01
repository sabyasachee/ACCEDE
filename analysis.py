import matplotlib.pyplot as plt
import numpy as np
from load import movies, load_output_movies

fw = open('output10.txt')
movie_valence_rmses, movie_arousal_rmses = range(30), range(30)
movie_valence_corrs, movie_arousal_corrs = range(30), range(30)
overall_valence_corrs, overall_arousal_corrs = range(11), range(11)
overall_valence_rmses, overall_arousal_rmses = range(11), range(11)
overall_valence_rmses[0] = 0.336
overall_arousal_rmses[0] = 0.440
overall_valence_corrs[0] = 0.515
overall_arousal_corrs[0] = 0.213
iteration = -1
flag = False
for line in fw:
	if line.startswith('*****************************'):
		iteration += 1
	elif 'new overall performance' in line:
		flag = True
	elif line.startswith('Valence rmse') and flag:
		parts = line.strip().split()
		rmse = float(parts[3])
		corr = float(parts[6])
		overall_valence_rmses[iteration + 1] = rmse
		overall_valence_corrs[iteration + 1] = corr
	elif line.startswith('Arousal rmse') and flag:
		parts = line.strip().split()
		rmse = float(parts[3])
		corr = float(parts[6])
		overall_arousal_rmses[iteration + 1] = rmse
		overall_arousal_corrs[iteration + 1] = corr
		flag = False
	elif line.startswith('Fold') and 'Valence rmse' in line:
		parts = line.strip().split()
		rmse = float(parts[5])
		corr = float(parts[8])
		movie = int(parts[1])
		if not iteration:
			movie_valence_rmses[movie] = []
			movie_valence_corrs[movie] = []
		movie_valence_rmses[movie].append(rmse)
		movie_valence_corrs[movie].append(corr)
	elif line.startswith('Fold') and 'Arousal rmse' in line:
		parts = line.strip().split()
		rmse = float(parts[5])
		corr = float(parts[8])
		movie = int(parts[1])
		if not iteration:
			movie_arousal_rmses[movie] = []
			movie_arousal_corrs[movie] = []
		movie_arousal_rmses[movie].append(rmse)
		movie_arousal_corrs[movie].append(corr)
fw.close()

mean_movie_valence_rmses = [np.mean(valence_rmses) for valence_rmses in movie_valence_rmses]
mean_movie_arousal_rmses = [np.mean(arousal_rmses) for arousal_rmses in movie_arousal_rmses]
mean_movie_valence_corrs = [np.mean(valence_corrs) for valence_corrs in movie_valence_corrs]
mean_movie_arousal_corrs = [np.mean(arousal_corrs) for arousal_corrs in movie_arousal_corrs]

print 'max valence corr:', movies[np.argmax(mean_movie_valence_corrs)]
print 'min valence corr:', movies[np.argmin(mean_movie_valence_corrs)]
print 'max arousal corr:', movies[np.argmax(mean_movie_arousal_corrs)]
print 'min arousal corr:', movies[np.argmin(mean_movie_arousal_corrs)]

print 

movie_valence_labels, movie_arousal_labels = load_output_movies()
mean_movie_valence_labels = [np.mean(valence_labels) for valence_labels in movie_valence_labels]
mean_movie_arousal_labels = [np.mean(arousal_labels) for arousal_labels in movie_arousal_labels]
print 'max valence :', movies[np.argmax(mean_movie_valence_labels)]
print 'min valence :', movies[np.argmin(mean_movie_valence_labels)]
print 'max arousal :', movies[np.argmax(mean_movie_arousal_labels)]
print 'min arousal :', movies[np.argmin(mean_movie_arousal_labels)]

# iterations = np.arange(10)
# for i in range(30):
# 	plt.plot(iterations, np.mean(movie_arousal_corrs[i])*np.ones(10))
# plt.show()
# plt.plot(iterations, overall_valence_rmses)
# plt.plot(iterations, overall_arousal_rmses)
# plt.show()
# plt.plot(iterations, overall_valence_corrs)
# plt.plot(iterations, overall_arousal_corrs)
# plt.show()