import load
import numpy
from sklearn.linear_model import BayesianRidge, ElasticNet, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

regressors = [BayesianRidge(), ElasticNet(), ElasticNetCV(max_iter = 10000), DecisionTreeRegressor(max_depth = 2)]
load.n_samples = 9800
feature_vectors = load.load_input()
valence_labels, arousal_labels, valence_correlations, arousal_correlations = load.load_output()
valence_features = load.sort_features(feature_vectors, valence_correlations)
arousal_features = load.sort_features(feature_vectors, arousal_correlations)
model_filenames = ["model_Bayesian", "model_ElasticNet", "model_ElasticNetCV", "model_DecisionTree"]

for regressor, model_filename in zip(regressors, model_filenames):
	print "Valence model : ", type(regressor).__name__
	regressor.fit(numpy.array(valence_features, dtype = "float"), numpy.array(valence_labels, dtype = "float"))
	filename = "valence_" + model_filename + ".pkl"
	joblib.dump(regressor, filename)

	print "Arousal model : ", type(regressor).__name__
	regressor.fit(numpy.array(arousal_features, dtype = "float"), numpy.array(arousal_labels, dtype = "float"))
	filename = "arousal_" + model_filename + ".pkl"
	joblib.dump(regressor, filename)