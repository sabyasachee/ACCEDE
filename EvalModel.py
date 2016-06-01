import numpy
import datetime 
import random 
import itertools
import sys 
import os 
import pdb
from sklearn import svm, grid_search, linear_model, cross_validation
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import LinearSVR as LSVR
from sklearn.svm import SVC
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GBC
from CreateArff import CreateArff
from GetWekaPredictions import GetWekaPredictions
import numpy as np

class BinaryEnsemble():
   def __init__(self, groups):
      self.groups = groups # Tuple of arrays with pairs of labels
      self.models = numpy.array([])

   def fit(self, train_features, train_labels, model_choice, model_params):
      train_label_groups = numpy.array([])
      self.models = numpy.array([])
      for group in self.groups:
         group_indices = numpy.array(len(train_labels)*[False])
         for item in group:
            group_indices = group_indices | (train_labels == item)
         train_labels_group = train_labels[group_indices]
         train_features_group = train_features[group_indices]
         sub_model = TrainModel(train_features_group, train_labels_group, model_choice, model_params)
         self.models = numpy.append(self.models, sub_model)

      return self
   
   def predict(self, test_features):
      all_preds = np.empty((len(test_features),len(self.models)))
      for i in range(len(self.models)):
         group = self.groups[i]
         model = self.models[i]
         preds = np.array(model.predict(test_features))
         indices_first_group_item = preds == group[0]
         preds[indices_first_group_item] = 0
         preds[numpy.logical_not(indices_first_group_item)] = 1
         all_preds[:,i] = preds

      # TODO - Implement different merging polices (ie. weighted average)

      #stacked_pred = numpy.zeros(test_features.shape[0])
      #power_coef = 1
      #for i in range(len(all_preds)):
      #   stacked_pred = stacked_pred + power_coef*all_preds[i].astype(int)
      #   power_coef = 2*power_coef

      #return stacked_pred

      return all_preds

class BagSVR():
   def __init__(self):
      self.SVRs = []

   def fit(self, train_features, train_labels, N, c_val=0.0001, tol_val=0.001):
      # break features into N sets
      feat_dim = train_features.shape[1]
      feat_per_bag = feat_dim / N
      self.SVRs = []
      for i in range(N):
         if i < N-1:
            cur_train_feat_bag = train_features[:,i*feat_per_bag:(i+1)*feat_per_bag]
         else:
            cur_train_feat_bag = train_features[:,i*feat_per_bag:]
         # now train individual SVR
         # model = svm.SVR(C=c_val, kernel='linear', tol=tol_val)
         # model = LSVR(C=c_val, tol=tol_val)
         model = SGDR(loss='epsilon_insensitive',alpha=c_val)
         print 'current training on dimensionality: ', cur_train_feat_bag.shape[1], '\n' 
         model.fit(cur_train_feat_bag, train_labels)
         self.SVRs.append(model)
      return self.SVRs

   def predict(self, test_features):
      feat_dim = test_features.shape[1]
      N = len(self.SVRs)
      feat_per_bag = feat_dim / N
      bagged_predictions = numpy.zeros((test_features.shape[0],1))
      for i in range(N):
         if i < N-1:
            cur_test_feat_bag = test_features[:,i*feat_per_bag:(i+1)*feat_per_bag]
         else:
            cur_test_feat_bag = test_features[:,i*feat_per_bag:]
         cur_model = self.SVRs[i]
         bagged_predictions = cur_model.predict(cur_test_feat_bag) 
      return bagged_predictions

class WekaSVR(): # training using SVR from weka
   def __init__(self, model_file = None, remove_model_file = True, force_model_filename = None):
      if model_file is None:
        self.model = '' 
      else:
        self.model = model_file
      self.temp_train_file = ''
      self.temp_test_file = ''
      self.temp_pred_file = ''
      self.remove_model_file = remove_model_file
      self.force_model_filename = force_model_filename

   def __del__(self):
      if self.model and self.remove_model_file == True:
         command = 'rm ' + self.model
         os.system(command) 
      if self.temp_train_file:
         command = 'rm ' + self.temp_train_file
         os.system(command) 
      if self.temp_test_file:
         command = 'rm ' + self.temp_test_file
         os.system(command)
      if self.temp_pred_file:
         command = 'rm ' + self.temp_pred_file
         os.system(command)

   def fit(self, train_features, train_labels, c_val=0.0001, L=0.1, weka_jar='weka-3-6-13/weka.jar'):
      # create temporary arff file using train features 
      temp_train_file = 'temp_train' + str(datetime.datetime.now()).split(' ')[1] + str(random.random()) + '.arff'
      if self.force_model_filename is None:
        model_file = 'temp_model' + str(datetime.datetime.now()).split(' ')[1] + str(random.random())
      else:
        model_file = self.force_model_filename
      CreateArff(temp_train_file, train_features, train_labels)
      command1 = 'java -Xmx4096m -classpath ' + weka_jar + ' weka.classifiers.meta.FilteredClassifier -v -o\
       -no-cv -c last -t "' + temp_train_file + '" -d "' + model_file + '" -F "weka.filters.unsupervised.attribute.Remove -R 1"\
       -W weka.classifiers.functions.SMOreg -- -C ' + str(c_val) + ' -N 1 -I \
       "weka.classifiers.functions.supportVector.RegSMOImproved -L ' + str(L) + \
       ' -W 1 -P 1.0E-12 -T 0.001 -V" -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0"'
      os.system(command1)
      self.temp_train_file = temp_train_file
      self.model = model_file
      return self.model 

   def predict(self, test_features, weka_jar='weka-3-6-13/weka.jar'):
      temp_test_file = 'temp_test' + str(datetime.datetime.now()).split(' ')[1] + str(random.random()) + '.arff'
      CreateArff(temp_test_file, test_features)
      model_file = self.model
      temp_pred_file = 'temp_pred_file' + str(datetime.datetime.now()).split(' ')[1] + str(random.random())
      command1 = 'java -Xmx4096m -classpath ' + weka_jar + '  weka.classifiers.meta.FilteredClassifier -o -c last -l "' \
      + model_file + '" -T "' + temp_test_file + '" -p 0 -distribution > "' + temp_pred_file + '"' 
      os.system(command1)
      predictions = GetWekaPredictions(temp_pred_file)
      self.temp_test_file = temp_test_file
      self.temp_pred_file = temp_pred_file
      return predictions

"""
Implementation of pairwise ranking using scikit-learn LinearSVC

Reference: 

    "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer 1999

    "Learning to rank from medical imaging data." Pedregosa, Fabian, et al., 
    Machine Learning in Medical Imaging 2012.


Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>

See also https://github.com/fabianp/pysofia for a more efficient implementation
of RankSVM using stochastic gradient descent methdos.
"""
def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)

        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.ravel()))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)
 

def TrainModel(train_features, train_labels, model_option, model_params) :
   if model_option == 'SVR':
      print 'Training SVR'
      c_val, tol_val, cache_size = model_params[0], model_params[1], model_params[2] 
      # model = svm.SVR(C=c_val, kernel='linear', tol=tol_val) # many more parameters here that can be tune like tolerance, kernel etc
      # model = LSVR(C=c_val, tol=tol_val) # linear implementation of SVR 
      model = SGDR(loss='epsilon_insensitive',alpha=c_val)
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'RF':
      print 'Training RF'
      n_est = int(model_params[0])
      model = RF(n_estimators=n_est)
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'LR':
      print 'Training LR'
      model = LR()
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'BagSVR':
      print 'Training BagSVR'
      N, c_val, tol_val = int(model_params[0]), model_params[1], model_params[2] 
      model = BagSVR()
      model.fit(train_features, train_labels, N, c_val, tol_val)
      return model
   elif model_option == 'SVC':
      print 'Training multi-class SVC'
      c_val, tol_val = model_params[0], model_params[1]
      model = svm.LinearSVC(tol=tol_val, C=c_val)
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'SVC-CV':
      print 'Training and optimizing multi-class SVC'
      c_val, tol_val, power_base = model_params[0], model_params[1], model_params[2] 
      model = svm.LinearSVC(tol=tol_val)
      c_params = []
      for i in range(-1,2):
        c_params.append(c_val*(power_base**i))
      parameters = {'C':c_params}
      grid_cv = grid_search.GridSearchCV(model, parameters)#, cv=len(train_labels)/10)
      grid_cv.fit(train_features, train_labels)
      return grid_cv
   elif model_option == 'kSVC':
      print 'Training multi-class kernel SVC'
      c_val, tol_val = model_params[0], model_params[1]
      model = svm.SVC(tol=tol_val, C=c_val)
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'WekaSVR':
      print 'Training Weka SVR'
      c_val, l_val = model_params[0], model_params[1]
      model = WekaSVR()
      model.fit(train_features, train_labels, c_val, l_val)
      return model
   elif model_option == 'RankSVM':
      print 'Training Rank SVM'
      model = RankSVM()
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'KNN':
      print 'Training K-Nearest Neighbors'
      neighbors = model_params[0]
      model = KNN(n_neighbors=neighbors)
      model.fit(train_features, train_labels)
      return model
   elif model_option == 'Boost':
      print 'Training Gradient Boosting Classifier'
      rate, N = model_params[0], int(model_params[1])
      model = GBC(learning_rate=rate,n_estimators=N)
      model.fit(train_features, train_labels)
      return model

def TestModel(model, test_features):
   return model.predict(test_features)
