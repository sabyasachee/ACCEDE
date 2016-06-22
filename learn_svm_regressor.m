function learn_svm_regressor(features_file_name, test_features_file_name, labels_file_name, test_labels_file_name)

	features = load(features_file_name);
	labels = load(labels_file_name);
	test_features = load(test_features_file_name);

	% disp(features)
	disp(labels)

	mdl = fitrsvm(features, labels);
	predictions = predict(mdl, test_features);
	dlmwrite(test_labels_file_name, predictions);