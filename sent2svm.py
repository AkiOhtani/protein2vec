#!/usr/bin/env python
# -*- coding: utf-8 -*-

from word2vec import Word2Vec
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import logging
import sys
import os
import time

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

model2 = Word2Vec.load_word2vec_format("amino_corpus_sent.txt.vec", binary=False)

i = 0
sent = {}
for f in open("amino_corpus_sent.txt", "rU"):
    sent[f.strip()] = i
    i += 1

preTrain = [f2.strip().split(',')[0] for f2 in open("../bio/derived_data/annotate_prot.txt", "rU")]
label = map(float, [f3.strip().split(',')[1] for f3 in open("../bio/derived_data/annotate_prot.txt", "rU")])

# zero_label = [float(0) for element in label]
# print label

train = []

for prot in preTrain:
    train.append(map(float, model2["sent_" + str(sent[prot])].tolist()))
# print train

# X = np.array(train, dtype=np.float).tolist()
# y = np.zeros_like(np.array(label, dtype=np.float)).tolist()

# print y
# time.sleep()

print "len(train[0]):", len(train[0])
print "len(train):", len(train)
print "len(label):", len(label)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, label, test_size=0.33, random_state=42)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
# lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
# print "svc:"+svc.score(X_test, y_test)
# print "rbf:"+rbf_svc.score(X_test, y_test)
# print "poly:"+poly_svc.score(X_test, y_test)
# print "linear:"+lin_svc.score(X_test, y_test)
# from sklearn import linear_model
# logReg = linear_model.LogisticRegression(C=1e5).fit(X_train, y_train)
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=8, n_init=10)

print """SVC(RBF Kernel)"""
scores = cross_validation.cross_val_score(rbf_svc, X_test, y_test, cv=10) # cvの回数はwindowの数と同じ
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from svm import *
from svmutil import *
'''
Examples of options: -s 0 -c 10 -t 1 -g 1 -r 1 -d 3
Classify a binary data with polynomial kernel (u'v+1)^3 and C = 10

options:
-s svm_type : set type of SVM (default 0)
    0 -- C-SVC
    1 -- nu-SVC
    2 -- one-class SVM
    3 -- epsilon-SVR
    4 -- nu-SVR
-t kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default 1/num_features)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.001)
-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

The k in the -g option means the number of attributes in the input data.
'''
# 学習
problem = svm_problem(y_train, X_train)
parameter = svm_parameter('-h 0 -s 0 -t 0')
t = svm_train(problem, parameter)

# 予測
result = svm_predict(y_test, X_test, t)

# print "[Result]"
# for r in result:
    # print r



# Set the parameters by cross-validation
print """###Support Vector Crassifier(SVC)###"""
"""
C : float, optional (default=1.0)
Penalty parameter C of the error term.
kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.
degree : int, optional (default=3)
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
gamma : float, optional (default=0.0)
Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is 0.0 then 1/n_features will be used instead.
coef0 : float, optional (default=0.0)
Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
probability: boolean, optional (default=False) :
Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
shrinking: boolean, optional (default=True) :
Whether to use the shrinking heuristic.
tol : float, optional (default=1e-3)
Tolerance for stopping criterion.
cache_size : float, optional
Specify the size of the kernel cache (in MB)
class_weight : {dict, ‘auto’}, optional
Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The ‘auto’ mode uses the values of y to automatically adjust weights inversely proportional to class frequencies.
verbose : bool, default: False
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
max_iter : int, optional (default=-1)
Hard limit on iterations within solver, or -1 for no limit.
random_state : int seed, RandomState instance, or None (default)
The seed of the pseudo random number generator to use when shuffling the data for probability estimation.
"""
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7], 'degree': [3, 4, 5, 6, 7, 8, 9, 10], 'C':[1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7], 'C':[1, 10, 100, 1000]}
                    # {'kernel': ['precomputed'], 'C':[1, 10, 100, 1000]}
                    ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)


print """###Logistic Regression Classifier(LRC)###"""
from sklearn import linear_model
"""
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
penalty : string, ‘l1’ or ‘l2’
Used to specify the norm used in the penalization.
dual : boolean
Dual or primal formulation. Dual formulation is only implemented for l2 penalty. Prefer dual=False when n_samples > n_features.
C : float, optional (default=1.0)
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
fit_intercept : bool, default: True
Specifies if a constant (a.k.a. bias or intercept) should be added the decision function.
intercept_scaling : float, default: 1
when self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased
class_weight : {dict, ‘auto’}, optional
Over-/undersamples the samples of each class according to the given weights. If not given, all classes are supposed to have weight one. The ‘auto’ mode selects weights inversely proportional to class frequencies in the training set.
random_state: int seed, RandomState instance, or None (default) :
The seed of the pseudo random number generator to use when shuffling the data.
tol: float, optional :
Tolerance for stopping criteria.
"""
tuned_parameters = [{'C': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]},
                    {'penalty': 'l1', 'C': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]},
                    {'penalty': 'l2', 'C': [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]}
                    ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(linear_model.LogisticRegression(C=1e5), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

print """###K-means###"""
from sklearn.cluster import KMeans
"""
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
n_clusters : int, optional, default: 8
The number of clusters to form as well as the number of centroids to generate.
max_iter : int, default: 300
Maximum number of iterations of the k-means algorithm for a single run.
n_init : int, default: 10
Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
init : {‘k-means++’, ‘random’ or an ndarray}
Method for initialization, defaults to ‘k-means++’:
‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
‘random’: choose k observations (rows) at random from data for the initial centroids.
If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
precompute_distances : boolean, default: True
Precompute distances (faster but takes more memory).
tol : float, default: 1e-4
Relative tolerance with regards to inertia to declare convergence
n_jobs : int, default: 1
The number of jobs to use for the computation. This works by breaking down the pairwise matrix into n_jobs even slices and computing them in parallel.
If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
random_state : integer or numpy.RandomState, optional
The generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global numpy random number generator.
"""
tuned_parameters = [{'init': 'k-means++', 'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'n_init': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                    {'init': 'random', 'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'n_init': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
                    ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KMeans(n_clusters=8, n_init=10), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

time.sleep()
