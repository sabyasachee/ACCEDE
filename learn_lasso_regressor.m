function learn_lasso_regressor(alpha, lambdas, features_file_name, labels_file_name, coeffs_file_name, info_file_name)

	% disp(features_file_name)
	% disp(labels_file_name)
	% disp(coeffs_file_name)
	% disp(alpha)
	% disp(lambdas)

	features = load(features_file_name);
	labels = load(labels_file_name);

	[coeffs, fitinfo] = lasso(features, labels, 'Alpha', alpha, 'Lambda', lambdas, 'Standardize', false);
	dlmwrite(coeffs_file_name, coeffs);
	dlmwrite(info_file_name, fitinfo.Intercept);
