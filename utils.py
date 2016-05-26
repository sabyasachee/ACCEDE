import numpy as np
import math

def get_statistics(values):
	# When all values equal zero, variance will be zero and kurtosis is inf. This is symbolised ny -1 as kurtosis value
	values = sorted(values[2:-2])
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
		lower_quartile = float(lower_values[quarter_length] + lower_values[quarter_length-1])/2
		upper_quartile = float(upper_values[quarter_length] + upper_values[quarter_length-1])/2
	else:
		lower_quartile = lower_values[quarter_length]
		upper_quartile = lower_values[quarter_length]
	_min = values[0]
	_max = values[length - 1]
	_range = _max - _min
	for value in values:
		squared = (value - mean)*(value - mean)
		double_squared = squared*squared
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
			values = list(np.load(folder + "/ACCEDE" + str(file_id).zfill(5) + "_" + suffix + ".npy"))
			mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range = get_statistics(values)
			fw.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (file_id, mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range))

def audio_statistics():
	n = 141
	with open("../results/audio.txt", "w") as fw:
		fw.write("file_id\tmfcc[1]\tmfcc[2]\tmfcc[3]\tmfcc[4]\tmfcc[5]\tmfcc[6]\tmfcc[7]\tmfcc[8]\tmfcc[9]\tmfcc[10]\tmfcc[11]\tmfcc[12]" +
			"\tpcm_LOGenergy\tmfcc_de[1]\tmfcc_de[2]\tmfcc_de[3]\tmfcc_de[4]\tmfcc_de[5]\tmfcc_de[6]\tmfcc_de[7]\tmfcc_de[8]\tmfcc_de[9]\tmfcc_de[10]" +
			"\tmfcc_de[11]\tmfcc_de[12]\tpcm_LOGenergy_de\tmfcc_de_de[1]\tmfcc_de_de[2]\tmfcc_de_de[3]\tmfcc_de_de[4]\tmfcc_de_de[5]\tmfcc_de_de[6]" +
			"\tmfcc_de_de[7]\tmfcc_de_de[8]\tmfcc_de_de[9]\tmfcc_de_de[10]\tmfcc_de_de[11]\tmfcc_de_de[12]\tpcm_LOGenergy_de_de\tvoiceProb\tHNR\tF0" +
			"\tvoiceProb_de\tHNR_de\tF0_de\tpcm_zcr\tpcm_zcr_de\tmfcc[1]_stddev\tmfcc[1]_amean\tmfcc[2]_stddev\tmfcc[2]_amean\tmfcc[3]_stddev\tmfcc[3]_amean" +
			"\tmfcc[4]_stddev\tmfcc[4]_amean\tmfcc[5]_stddev\tmfcc[5]_amean\tmfcc[6]_stddev\tmfcc[6]_amean\tmfcc[7]_stddev\tmfcc[7]_amean\tmfcc[8]_stddev" +
			"\tmfcc[8]_amean\tmfcc[9]_stddev\tmfcc[9]_amean\tmfcc[10]_stddev\tmfcc[10]_amean\tmfcc[11]_stddev\tmfcc[11]_amean\tmfcc[12]_stddev\tmfcc[12]_amean" +
			"\tpcm_LOGenergy_stddev\tpcm_LOGenergy_amean\tmfcc_de[1]_stddev\tmfcc_de[1]_amean\tmfcc_de[2]_stddev\tmfcc_de[2]_amean\tmfcc_de[3]_stddev" +
			"\tmfcc_de[3]_amean\tmfcc_de[4]_stddev\tmfcc_de[4]_amean\tmfcc_de[5]_stddev\tmfcc_de[5]_amean\tmfcc_de[6]_stddev\tmfcc_de[6]_amean\tmfcc_de[7]_stddev" +
			"\tmfcc_de[7]_amean\tmfcc_de[8]_stddev\tmfcc_de[8]_amean\tmfcc_de[9]_stddev\tmfcc_de[9]_amean\tmfcc_de[10]_stddev\tmfcc_de[10]_amean\tmfcc_de[11]_stddev" +
			"\tmfcc_de[11]_amean\tmfcc_de[12]_stddev\tmfcc_de[12]_amean\tpcm_LOGenergy_de_stddev\tpcm_LOGenergy_de_amean\tmfcc_de_de[1]_stddev\tmfcc_de_de[1]_amean" +
			"\tmfcc_de_de[2]_stddev\tmfcc_de_de[2]_amean\tmfcc_de_de[3]_stddev\tmfcc_de_de[3]_amean\tmfcc_de_de[4]_stddev\tmfcc_de_de[4]_amean\tmfcc_de_de[5]_stddev" +
			"\tmfcc_de_de[5]_amean\tmfcc_de_de[6]_stddev\tmfcc_de_de[6]_amean\tmfcc_de_de[7]_stddev\tmfcc_de_de[7]_amean\tmfcc_de_de[8]_stddev\tmfcc_de_de[8]_amean" +
			"\tmfcc_de_de[9]_stddev\tmfcc_de_de[9]_amean\tmfcc_de_de[10]_stddev\tmfcc_de_de[10]_amean\tmfcc_de_de[11]_stddev\tmfcc_de_de[11]_amean\tmfcc_de_de[12]_stddev" +
			"\tmfcc_de_de[12]_amean\tpcm_LOGenergy_de_de_stddev\tpcm_LOGenergy_de_de_amean\tvoiceProb_stddev\tvoiceProb_amean\tHNR_stddev\tHNR_amean\tF0_stddev\tF0_amean" +
			"\tvoiceProb_de_stddev\tvoiceProb_de_amean\tHNR_de_stddev\tHNR_de_amean\tF0_de_stddev\tF0_de_amean\tpcm_zcr_stddev\tpcm_zcr_amean\tpcm_zcr_de_stddev" +
			"\tpcm_zcr_de_amean\n[mean\tmedian\tstd\tkurtosis\tlower_quartile\tupper_quartile\tmin\tmax\trange]\n")
		means = []
		medians = []
		stds = []
		kurtosises = []
		lower_quartiles = []
		upper_quartiles = []
		mins = []
		maxs = []
		ranges = []
		for file_id in range(0, 9800):
			filename = "../audio_results/ACCEDE" + str(file_id).zfill(5) + "_features.txt"
			print "Working on audio_file", filename 
			feature_vectors = []
			with open(filename) as fr:
				for i in range(0, 149):
					line = fr.readline()
				for line in fr:
					feature_vector = [float(value) for value in line.strip().split(",")[2:-1]]
					feature_vectors.append(feature_vector)
			length = len(feature_vectors)
			dimension = len(feature_vectors[0])
			print "\tNumber of vectors(x) = ", length
			print "\tNumber of features(D) = ", dimension
			for i in range(0, dimension):
				values = [feature_vector[i] for feature_vector in feature_vectors]
				mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range = get_statistics(values)
				means.append(mean)
				medians.append(median)
				stds.append(std)
				kurtosises.append(kurtosis)
				lower_quartiles.append(lower_quartile)
				upper_quartiles.append(upper_quartile)
				mins.append(_min)
				maxs.append(_max)
				ranges.append(_range)
			print "\tWriting to output"
			fw.write("%d\t" % (file_id))
			for i in range(0, dimension):
				fw.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t" % (means[i], medians[i], stds[i], kurtosises[i], lower_quartiles[i], upper_quartiles[i],
					 mins[i], maxs[i], ranges[i]))
			fw.write("\n")

# statistics("../results/intensity", "intensity")
# statistics("../results/luminance", "luma")
# statistics("../results/optical_flow", "flow")
audio_statistics()