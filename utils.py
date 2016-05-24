import numpy as np
import math

def statistics(folder, suffix):
	with open("../results/" + suffix + ".txt", "w") as fw:
		fw.write("file_id\tmean\tmedian\tstd\tkurtosis\tlower_quartile\tupper_quartile\tmin\tmax\trange\n")
		for file_id in range(0, 9800):
			values = list(np.load(folder + "/ACCEDE" + str(file_id).zfill(5) + "_" + suffix + ".npy"))
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
			kurtosis = float(fourth_moment)/(variance*variance)
			std = math.sqrt(variance)
			fw.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (file_id, mean, median, std, kurtosis, lower_quartile, upper_quartile, _min, _max, _range))

# statistics("../results/intensity", "intensity")
statistics("../results/luminance", "luma")
statistics("../results/optical_flow", "flow")