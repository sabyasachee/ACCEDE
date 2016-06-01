import numpy
import math

def get_statistics(values, clip = True):
	# When all values equal zero, variance will be zero and kurtosis is inf. This is symbolised by -1 as kurtosis value
	if clip:
		values = sorted(values[2:-2])
	else:
		values = sorted(values)
	length = len(values)
	half_length = length/2
	quarter_length = half_length/2
	mean = float(sum(values))/length
	if not length%2:
		# even length
		median = float(values[half_length] + values[half_length-1])/2
		lower_values = values[:half_length]
		upper_values = values[half_length:]
	else:
		# odd length
		median = values[half_length]
		lower_values = values[:half_length]
		upper_values = values[half_length + 1:]
	if not half_length%2:
		# even half length
		lower_quartile = float(lower_values[quarter_length] + lower_values[quarter_length-1])/2
		upper_quartile = float(upper_values[quarter_length] + upper_values[quarter_length-1])/2
	else:
		lower_quartile = lower_values[quarter_length]
		upper_quartile = upper_values[quarter_length]
	_min = values[0]
	_max = values[length - 1]
	_range = _max - _min
	squared = 0
	double_squared = 0
	for value in values:
		squared += (value - mean)*(value - mean)
		double_squared += (value - mean)*(value - mean)*(value - mean)*(value - mean)
	fourth_moment = float(double_squared)/length
	variance = float(squared)/length
	if variance > 0.0:
		kurtosis = float(fourth_moment)/(variance*variance)
	else:
		kurtosis = -1.0
	std = math.sqrt(variance)

	return mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range

def statistics(folder, suffix):
	with open("../results/" + suffix + ".txt", "w") as fw:
		fw.write("file_id\tmean\tmedian\tstd\tkurtosis\tlower_quartile\tupper_quartile\tmin\tmax\trange\n")
		for file_id in range(0, 9800):
			values = list(numpy.load(folder + "/ACCEDE" + str(file_id).zfill(5) + "_" + suffix + ".npy"))
			mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range = get_statistics(values)
			fw.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (file_id, mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range))

def audio_statistics():
	audio_features = ["mfcc[1]","mfcc[2]","mfcc[3]","mfcc[4]","mfcc[5]","mfcc[6]","mfcc[7]","mfcc[8]","mfcc[9]","mfcc[10]","mfcc[11]","mfcc[12]",
	"pcm_LOGenergy","mfcc_de[1]","mfcc_de[2]","mfcc_de[3]","mfcc_de[4]","mfcc_de[5]","mfcc_de[6]","mfcc_de[7]","mfcc_de[8]","mfcc_de[9]","mfcc_de[10]",
	"mfcc_de[11]","mfcc_de[12]","pcm_LOGenergy_de","mfcc_de_de[1]","mfcc_de_de[2]","mfcc_de_de[3]","mfcc_de_de[4]","mfcc_de_de[5]","mfcc_de_de[6]",
	"mfcc_de_de[7]","mfcc_de_de[8]","mfcc_de_de[9]","mfcc_de_de[10]","mfcc_de_de[11]","mfcc_de_de[12]","pcm_LOGenergy_de_de","voiceProb","HNR","F0",
	"voiceProb_de","HNR_de","F0_de","pcm_zcr","pcm_zcr_de","mfcc[1]_stddev","mfcc[1]_amean","mfcc[2]_stddev","mfcc[2]_amean","mfcc[3]_stddev","mfcc[3]_amean",
	"mfcc[4]_stddev","mfcc[4]_amean","mfcc[5]_stddev","mfcc[5]_amean","mfcc[6]_stddev","mfcc[6]_amean","mfcc[7]_stddev","mfcc[7]_amean","mfcc[8]_stddev",
	"mfcc[8]_amean","mfcc[9]_stddev","mfcc[9]_amean","mfcc[10]_stddev","mfcc[10]_amean","mfcc[11]_stddev","mfcc[11]_amean","mfcc[12]_stddev","mfcc[12]_amean",
	"pcm_LOGenergy_stddev","pcm_LOGenergy_amean","mfcc_de[1]_stddev","mfcc_de[1]_amean","mfcc_de[2]_stddev","mfcc_de[2]_amean","mfcc_de[3]_stddev",
	"mfcc_de[3]_amean","mfcc_de[4]_stddev","mfcc_de[4]_amean","mfcc_de[5]_stddev","mfcc_de[5]_amean","mfcc_de[6]_stddev","mfcc_de[6]_amean",
	"mfcc_de[7]_stddev","mfcc_de[7]_amean","mfcc_de[8]_stddev","mfcc_de[8]_amean","mfcc_de[9]_stddev","mfcc_de[9]_amean","mfcc_de[10]_stddev",
	"mfcc_de[10]_amean","mfcc_de[11]_stddev","mfcc_de[11]_amean","mfcc_de[12]_stddev","mfcc_de[12]_amean","pcm_LOGenergy_de_stddev","pcm_LOGenergy_de_amean",
	"mfcc_de_de[1]_stddev","mfcc_de_de[1]_amean","mfcc_de_de[2]_stddev","mfcc_de_de[2]_amean","mfcc_de_de[3]_stddev","mfcc_de_de[3]_amean","mfcc_de_de[4]_stddev",
	"mfcc_de_de[4]_amean","mfcc_de_de[5]_stddev","mfcc_de_de[5]_amean","mfcc_de_de[6]_stddev","mfcc_de_de[6]_amean","mfcc_de_de[7]_stddev","mfcc_de_de[7]_amean",
	"mfcc_de_de[8]_stddev","mfcc_de_de[8]_amean","mfcc_de_de[9]_stddev","mfcc_de_de[9]_amean","mfcc_de_de[10]_stddev","mfcc_de_de[10]_amean","mfcc_de_de[11]_stddev",
	"mfcc_de_de[11]_amean","mfcc_de_de[12]_stddev","mfcc_de_de[12]_amean","pcm_LOGenergy_de_de_stddev","pcm_LOGenergy_de_de_amean","voiceProb_stddev","voiceProb_amean",
	"HNR_stddev","HNR_amean","F0_stddev","F0_amean","voiceProb_de_stddev","voiceProb_de_amean","HNR_de_stddev","HNR_de_amean","F0_de_stddev","F0_de_amean",
	"pcm_zcr_stddev","pcm_zcr_amean","pcm_zcr_de_stddev","pcm_zcr_de_amean"]
	n_features = len(audio_features)
	for i in range(0, n_features):
		print audio_features[i]
		with open("../results/" + audio_features[i] + ".txt", "w") as fw:
			fw.write("file_id\tmean\tmedian\tstd\tkurtosis\tlower_quartile\tupper_quartile\tmin\tmax\trange\n")
			for file_id in range(0, 9800):
				with open("../audio_results/ACCEDE" + str(file_id).zfill(5) + "_features.txt") as fr:
					for j in range(0, 149):
						line = fr.readline()
					array = []
					for line in fr:
						values = line.strip().split(",")[2:-1]
						array.append(float(values[i]))
					# print array
					# print len(array)
					if not file_id%1000:
						print "\t", file_id
					mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range = get_statistics(array)
					fw.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (file_id, mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range))

def save_correlation():
	header =	[
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
				"pcm_zcr","pcm_zcr_de","pcm_zcr_stddev","pcm_zcr_amean","pcm_zcr_de_stddev","pcm_zcr_de_amean"
				]

	arousal_values = []
	valence_values = []
	valence_variance_values = []
	arousal_variance_values = []
	with open("../annotations/ACCEDEranking.txt") as fr:
		line = fr.readline()
		for i in range(0, 9800):
			values = fr.readline().strip().split("\t")[4:]
			true_values = [float(value) for value in values]
			valence_value, arousal_value, valence_variance, arousal_variance = true_values
			valence_values.append(valence_value)
			arousal_values.append(arousal_value)
			valence_variance_values.append(valence_variance)
			arousal_variance_values.append(arousal_variance)
	fw1 = open("../results/valence_correlations.txt", "w")
	fw2 = open("../results/arousal_correlations.txt", "w")
	fw1.write("feature_name\tmean\tmedian\tstd\tkurtosis\tlower_quartile\tupper_quartile\tmin\tmax\trange\n")
	fw2.write("feature_name\tmean\tmedian\tstd\tkurtosis\tlower_quartile\tupper_quartile\tmin\tmax\trange\n")
	for feature_name in header:
		mean_vectors = []
		median_vectors = []
		std_vectors = []
		kurt_vectors = []
		lower_quartile_vectors = []
		upper_quartile_vectors = []
		min_vectors = []
		max_vectors = []
		range_vectors = []
		with open("../results/" + feature_name + ".txt") as fr:
			line = fr.readline()
			for i in range(0, 9800):
				line = fr.readline()
				values = line.strip().split("\t")[1:]
				mean, median, std, kurt, lower_quartile, upper_quartile, _min, _max, _range = [float(value) for value in values]
				mean_vectors.append(mean)
				median_vectors.append(median)
				std_vectors.append(std)
				kurt_vectors.append(kurt)
				lower_quartile_vectors.append(lower_quartile)
				upper_quartile_vectors.append(upper_quartile)
				min_vectors.append(_min)
				max_vectors.append(_max)
				range_vectors.append(_range)
		valence_matrix = [valence_values, mean_vectors, median_vectors, std_vectors, kurt_vectors, lower_quartile_vectors, upper_quartile_vectors, 
			min_vectors, max_vectors, range_vectors]
		arousal_matrix = [arousal_values, mean_vectors, median_vectors, std_vectors, kurt_vectors, lower_quartile_vectors, upper_quartile_vectors, 
			min_vectors, max_vectors, range_vectors]
		valence_correlations = numpy.corrcoef(valence_matrix)[0]
		arousal_correlations = numpy.corrcoef(arousal_matrix)[0]
		fw1.write("%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (feature_name, valence_correlations[1], valence_correlations[2], valence_correlations[3],
																				valence_correlations[4],  valence_correlations[5], valence_correlations[6],
																				valence_correlations[7],  valence_correlations[8], valence_correlations[9]))
		fw2.write("%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (feature_name, arousal_correlations[1], arousal_correlations[2], arousal_correlations[3],
																				arousal_correlations[4],  arousal_correlations[5], arousal_correlations[6],
																				arousal_correlations[7],  arousal_correlations[8], arousal_correlations[9]))
		print feature_name, "done"

	fw1.close()
	fw2.close()

# statistics("../results/intensity", "intensity")
# statistics("../results/luminance", "luma")
# statistics("../results/optical_flow", "flow")
# audio_statistics()
# save_correlation()
