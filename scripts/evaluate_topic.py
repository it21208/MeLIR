# !/usr/bin/python
# author = Alexandros Ioannidis
import lightgbm as lgb
from lightgbm import LGBMRegressor
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

#uncomment the import of the os library in case you need to install scikit-learn
#import os
#os.system('pip install scikit-learn')


def evaluate_topic(X_train, y_train, X_test, classifier):

    if classifier == 'svm':
        # Create a linear SVM classifier. Instead of subsampling I change the value of the 'class_weight' parameter to 'balanced'
        clf = svm.SVC(kernel='linear', class_weight=None, probability=True)  # decision_function_shape='ovo'
        clf.fit(X_train, y_train)
        new_p, new_y_test_predict_log_proba = ([] for i in range(2))
        
        #for val in clf.decision_function(X_test):
        #    new_p.append(val - min(clf.decision_function(X_test)))
        
        for val in clf.predict_log_proba(X_test)[:, 1]:
            new_y_test_predict_log_proba.append(math.exp(val))
        
        y_test = clf.predict_proba(X_test)[:, 1] # The correct is with 1 apparently  #y_test = clf.predict_proba(X_test)[:, 0]

        new_p = new_y_test_predict_log_proba
        return(y_test, new_p)

    elif classifier == 'lgb':
        regr = LGBMRegressor(boosting_type='gbdt')
        regr.fit(X_train, y_train)
        y_test = regr.predict(X_test)
        ''' if you want to use LGBMClassifier ucomment the lines below '''
        #train_data = lgb.Dataset(X_train, label=y_train)
        #clf = lgb.train(param, train_data)
        #y_test = clf.predict(X_test)
        #new_p = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1] # supported only for LGBMClassifier
        new_p = None
        return(y_test, new_p)

    elif classifier == 'lsvr':
        regr = LinearSVR(random_state=0, tol=1e-5)
        regr.fit(X_train, y_train)
        y_test = regr.predict(X_test)
        return(y_test, None)

    elif classifier == 'svr':
        #svr = SVR(kernel='rbf', C=math.pow(2,C), gamma=math.pow(2,gamma), cache_size=2000, verbose=False, max_iter=-1, shrinking=False)
        regr = svm.SVR(C=1.0, epsilon=0.2, probability=True)
        regr.fit(X_train, y_train)
        y_test = regr.predict(X_test)
        return(y_test, None)

    elif classifier == 'sgd':
        clf = SGDClassifier(loss='hinge', penalty='l2',
                            alpha=1e-3, random_state=42, max_iter=5, tol=None)
        clf.fit(X_train, y_train)
        y_test = clf.predict(X_test)
        return(y_test, None)

    elif classifier == 'xgb':
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        y_test = clf.predict(X_test)
        return(y_test, None)

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
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=2, random_state=0)
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
        X_train_sampled, y_train_sampled = smt.fit_sample( X_train.toarray(), np.asarray(y_train))
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
