from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from cross_val import simple_cv 
import load
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression, ARDRegression, BayesianRidge, ElasticNet, ElasticNetCV, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

load.n_samples = 9800
feature_vectors = load.load_input()
valence_labels, arousal_labels, valence_correlations, arousal_correlations = load.load_output()
for i in [1., 0.9, 0.8, 0.5, 0.1, 0.01, 0.001]:
	n_dimension = int(load.n_features*i*9)
	print "n_dimension", n_dimension
	valence_features = load.sort_features(feature_vectors, valence_correlations, n_dimension = n_dimension)
	arousal_features = load.sort_features(feature_vectors, arousal_correlations, n_dimension = n_dimension)
	regressors = [DecisionTreeRegressor(max_depth = 2), DecisionTreeRegressor(max_depth = 5), DecisionTreeRegressor(max_depth = 10)]
	regressors_1 = [LinearRegression(), 
				BayesianRidge(), 
				ElasticNet(), 
				ElasticNetCV(max_iter = 10000), 
				Lasso(), 
				KNeighborsRegressor(), 
				DecisionTreeRegressor(max_depth = 2), 
				DecisionTreeRegressor(max_depth = 5), 
				DecisionTreeRegressor(max_depth = 10), 
				RandomForestRegressor(), 
				AdaBoostRegressor()
				]
	print "Valence"
	print simple_cv(valence_features, valence_labels, regressors_1)
	print "Arousal"
	print simple_cv(arousal_features, valence_labels, regressors_1)