import csv
from load import movies

filename = 'output22.txt'
outfilename = 'Boosting_Linear_50.csv'
with open(outfilename, 'w') as fw:
	writer = csv.writer(fw, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
	with open(filename) as fr:
		iteration = 0
		valence_baseline_rmse, valence_baseline_corr, valence_baseline_ccc = None, None, None
		valence_movies_rmse, valence_movies_corr = range(len(movies)), range(len(movies))
		valence_overall_rmse, valence_overall_corr, valence_overall_ccc, valence_best_shift = [], [], [], []
		
		arousal_baseline_rmse, arousal_baseline_corr, arousal_baseline_ccc = None, None, None
		arousal_movies_rmse, arousal_movies_corr = range(len(movies)), range(len(movies))
		arousal_overall_rmse, arousal_overall_corr, arousal_overall_ccc, arousal_best_shift = [], [], [], []
		
		header = []
		for movie in movies:
			header.append(movie + '_valence_rmse')
			header.append(movie + '_valence_corr')
			header.append(movie + '_arousal_rmse')
			header.append(movie + '_arousal_corr')
		writer.writerow(header)
		for line in fr:
			parts = line.strip().split()
			if iteration == 0 and 'Valence rmse' in line:
				valence_baseline_rmse = float(parts[3])
				valence_baseline_corr = float(parts[6])
				valence_baseline_ccc = float(parts[9])
			if iteration == 0 and 'Arousal rmse' in line:
				arousal_baseline_rmse = float(parts[3])
				arousal_baseline_corr = float(parts[6])
				arousal_baseline_ccc = float(parts[9])
			if '**************' in line:
				if iteration:
					row = []
					for valence_rmse, valence_corr, arousal_rmse, arousal_corr in zip(valence_movies_rmse, 
						valence_movies_corr, arousal_movies_rmse, arousal_movies_corr):
						row.extend([valence_rmse, valence_corr, arousal_rmse, arousal_corr])
					writer.writerow(row)
				iteration += 1
			if 'best valence shift' in line:
				valence_best_shift.append(int(parts[3]))
			if 'best arousal shift' in line:
				arousal_best_shift.append(int(parts[3]))	
			if iteration and 'Fold' in line and 'Valence rmse' in line:
				index = int(parts[1])
				rmse = float(parts[5])
				corr = float(parts[8])
				valence_movies_rmse[index] = rmse
				valence_movies_corr[index] = corr
			if iteration and 'Fold' in line and 'Arousal rmse' in line:
				index = int(parts[1])
				rmse = float(parts[5])
				corr = float(parts[8])
				arousal_movies_rmse[index] = rmse
				arousal_movies_corr[index] = corr
			if iteration and 'Fold' not in line and 'Valence rmse' in line and 'ccc' in line:
				rmse = float(parts[3])
				corr = float(parts[6])
				ccc = float(parts[9])
				valence_overall_rmse.append(rmse)
				valence_overall_corr.append(corr)
				valence_overall_ccc.append(ccc)
			if iteration and 'Fold' not in line and 'Arousal rmse' in line and 'ccc' in line:
				rmse = float(parts[3])
				corr = float(parts[6])
				ccc = float(parts[9])
				arousal_overall_rmse.append(rmse)
				arousal_overall_corr.append(corr)
				arousal_overall_ccc.append(ccc)

		iteration -= 1
		writer.writerow(['VALENCE','rmse','corr','ccc', 'shift'])
		writer.writerow(['h0', valence_baseline_rmse, valence_baseline_corr, valence_baseline_ccc])
		for i in range(iteration):
			writer.writerow(['h%d' % (i + 1), valence_overall_rmse[i], valence_overall_corr[i], 
				valence_overall_ccc[i], valence_best_shift[i]])
		writer.writerow(['AROUSAL','rmse','corr','ccc', 'shift'])
		writer.writerow(['h0', arousal_baseline_rmse, arousal_baseline_corr, arousal_baseline_ccc])
		for i in range(iteration):
			writer.writerow(['h%d' % (i + 1), arousal_overall_rmse[i], arousal_overall_corr[i], 
				arousal_overall_ccc[i], arousal_best_shift[i]])

print valence_baseline_rmse, valence_baseline_corr, valence_baseline_ccc
print arousal_baseline_rmse, arousal_baseline_corr, arousal_baseline_ccc
print iteration