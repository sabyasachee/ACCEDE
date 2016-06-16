import math

def save_feature_header_videos():
	with open('../valence_correlations_videos.txt', 'w') as fw:
		with open('../results/valence_correlations.txt') as fr:
			header = fr.readline()
			correlations = []
			for line in fr:
				values = line.strip().split('\t')
				feature = values[0]
				statistics = [float(value) for value in values[1:]]
				for i, statistic in enumerate(statistics):
					if math.isnan(statistic):
						statistic = -10
					correlations.append((feature, i, statistic))
			correlations = sorted(correlations, key = lambda correlation: correlation[2], reverse = True)
			for feature, statistic, value in correlations:
				fw.write('%s\t%d\t%f\n' % (feature, statistic, value))

	with open('../arousal_correlations_videos.txt', 'w') as fw:
		with open('../results/arousal_correlations.txt') as fr:
			header = fr.readline()
			correlations = []
			for line in fr:
				values = line.strip().split('\t')
				feature = values[0]
				statistics = [float(value) for value in values[1:]]
				for i, statistic in enumerate(statistics):
					if math.isnan(statistic):
						statistic = -10
					correlations.append((feature, i, statistic))
			correlations = sorted(correlations, key = lambda correlation: correlation[2], reverse = True)
			for feature, statistic, value in correlations:
				fw.write('%s\t%d\t%f\n' % (feature, statistic, value))

save_feature_header_videos()
