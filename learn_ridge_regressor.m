function learn_ridge_regressor(alpha, features_file_name, labels_file_name, coeffs_file_name)

	features = load(features_file_name);
	lables = load(labels_file_name);

	coeffs = ridge(lables, features, alpha, 0);
	dlmwrite(coeffs_file_name, coeffs);
