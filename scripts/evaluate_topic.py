# author = Alexandros Ioannidis
import lightgbm as lgb
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from rlscore.learner import CGRLS
from rlscore.measure import auc
from sklearn.svm import LinearSVR
# SVR for sparse matrices (csr)
#from sklearn.svm.sparse import SVC
import numpy as np
import math

''' uncomment the import of the os library in case you need to install scikit-learn'''
#import os 
#os.system('pip install scikit-learn')

def evaluate_topic(X_train, y_train, X_test, classifier):

  if classifier == 'svm':
    # Create a linear SVM classifier. Instead of subsampling I change the value of the 'class_weight' parameter to 'balanced'
    clf = svm.SVC(kernel='linear', class_weight=None, probability=True)  # decision_function_shape='ovo'
    #kernels = ['linear', 'rbf', 'poly']  #clf = svm.SVC(kernel='rbf')
    #clf = svm.SVC(C=1.0, kernel='linear', degree=3, probability=True, gamma='auto')
    clf.fit(X_train, y_train)
    p = clf.decision_function(X_test)
    temp_min_val = min(p)
    new_p = []
    for val in p:
      new_p.append(val - temp_min_val)
    #pred_vals = clf.predict(X_test)
    y_test_predict_log_proba = clf.predict_log_proba(X_test)[:,1]
    new_y_test_predict_log_proba = []
    for val in y_test_predict_log_proba:
      new_y_test_predict_log_proba.append(math.exp(val))
    # the correct is with 1 apparently
    y_test = clf.predict_proba(X_test)[:, 1]
    #y_test = clf.predict_proba(X_test)[:, 0]
    #pred_vals_own = np.array([predict(x) for x in dec_vals])
    new_p = new_y_test_predict_log_proba
    return(y_test, new_p)

  elif classifier == 'lgb':
    param = {'num_leaves': 31, 'num_iterations': 100, 'max_depth': -1, 'objective': 'binary', 'is_unbalance': True, 'metric': ['l2', 'binary_logloss'], 'verbose': -1}
    #param = {'num_leaves': 31, 'num_iterations': 100, 'max_depth': -1, 'objective': 'binary', 'is_unbalance': False, 'metric': ['l2','binary_logloss'], 'verbose': -1 }
    #param = { 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': 0  }
    train_data = lgb.Dataset(X_train, label=y_train)
    clf = lgb.train(param, train_data)  
    y_test = clf.predict(X_test)
    return(y_test, None)

  elif classifier == 'lsvr':
    # example of SVR declaration 
    #svr = SVR(kernel='rbf', C=math.pow(2,C), gamma=math.pow(2,gamma), cache_size=2000, verbose=False, max_iter=-1, shrinking=False)
    #class sklearn.svm.LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
    regr = LinearSVR(random_state=0, tol=1e-5) 
    regr.fit(X_train, y_train)
    # predict(T) - This function does classification or regression on an array of test vectors T.
    # predict_proba(T) - This function does classification or regression on a test vector T given a model with probability information.
    # Returns the probability of the sample for each class in the model, where classes are ordered by arithmetical order.
    # The probability model is created using cross validation, so the results can be slightly different than those obtained by predict. Also, it will be meaningless results on very small datasets.
    # score(X, y) -  Returns the explained variance of the prediction
    y_test = regr.predict(X_test)
    return(y_test, None)

  elif classifier == 'svr':
    #svr = SVR(kernel='rbf', C=math.pow(2,C), gamma=math.pow(2,gamma), cache_size=2000, verbose=False, max_iter=-1, shrinking=False)
    #class sklearn.svm.LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
    regr = svm.SVR(C=1.0, epsilon=0.2, probability=True)
    regr.fit(X_train, y_train)
    # predict(T) - This function does classification or regression on an array of test vectors T.
    # predict_proba(T) - This function does classification or regression on a test vector T given a model with probability information.
    # Returns the probability of the sample for each class in the model, where classes are ordered by arithmetical order.
    # The probability model is created using cross validation, so the results can be slightly different than those obtained by predict. 
    # Also, it will be meaningless results on very small datasets.  score(X, y) Returns the explained variance of the prediction
    y_test = regr.predict(X_test)
    return(y_test, None)

  elif classifier == 'sgd':
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return(y_test, None)

  elif classifier == 'xgb':
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return(y_test, None)

  # CGRLS does not support multi-output learning, so we train one classifier for the first column of Y. Multi-class learning 
  # would be implemented by training one CGRLS for each column, and taking the argmax of class predictions.
  elif classifier == 'rls':
    #clf = CGRLS(X_train, Y_train[:, 0], regparam=100.0)
    clf = CGRLS(X_train, y_train, regparam=100.0)
    y_test = clf.predict(X_test)
    return(y_test, None)

  elif classifier == 'dtc':
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return(y_test, None)

  elif classifier == 'rfc':
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return(y_test, None)

  elif classifier == 'kne':
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return(y_test, None)     

  elif classifier == 'smote':
    smt = SMOTE()
    smt = SMOTE(sampling_strategy='auto')
    X_train_sampled, y_train_sampled = smt.fit_sample(X_train.toarray(), np.asarray(y_train))
    clf = LinearSVC().fit(X_train_sampled, y_train_sampled)
    y_test = clf.predict(X_test) 
    return(y_test, None)

  elif classifier == 'lr':
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return(y_test, None)
  
  else:
    raise Exception('Wrong classifier')

  return(y_test, None)
