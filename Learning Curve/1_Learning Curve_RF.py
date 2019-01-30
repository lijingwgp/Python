# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:07:17 2019

@author: jing.o.li
"""

############################
##### Data Preparation #####
############################

import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

file = open("Funnel_Final.pickle",'rb')
funnel = pickle.load(file)
file.close()

x = funnel.drop(['LEAD_STATUS'],axis=1)
y = funnel.LEAD_STATUS
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y)



################################
##### Baseline Performance #####
################################

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
roc_auc_score(y_test, y_pred)



##########################
##### Learning Curve #####
##########################

# n_estimators
# It represents the number of trees in the forest. 

n_estimators = [100, 200, 500, 1000, 2000]
train_results = []
test_results = []

for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1).fit(x_train,y_train)
    train_pred = rf.predict(x_train)
    auc = roc_auc_score(y_train, train_pred)
    train_results.append(auc)
    
    test_pred = rf.predict(x_test)
    auc = roc_auc_score(y_test, test_pred)
    test_results.append(auc)

from matplotlib.legend_handler import HandlerLind2D
line1, = plt.plot(n_estimators, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(n_estimators, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘n_estimators’)
plt.show()

# max_depth
# It represents the depth of each tree in the forest. The deeper the tree, the more
# splits it has and it captures more information about the data.
# 1-32 usually is a good range to start with, but since our dataset is not normalized
# nor scaled, so we might need a bigger range.

max_depths = np.linspace(1, 32, 168, endpoint=True)
train_results = []
test_results = []

for each in max_depths:
    rf = RandomForestClassifier(max_depth=each, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    auc = roc_auc_score(y_train, train_pred)
    train_results.append(auc)
    
    test_pred = rf.predict(x_test)
    auc = roc_auc_score(y_test, test_pred)
    test_results.append(auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(max_depths, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘Tree depth’)
plt.show()

# min_samples_split
# min_samples_split represents the minimum number of samples required to split
# an internal node. This can vary between considering at least one sample
# at each node to considering all of the samples at each node. 
#
# When we increase this parameter, each tree in the forest becomes more constrained
# as it has to consider more samples at each node. 

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []

for each in min_samples_splits:
    rf = RandomForestClassifier(min_samples_splits=each, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    auc = roc_auc_score(y_train, train_pred)
    train_results.append(auc)
    
    test_pred = rf.predict(x_test)
    auc = roc_auc_score(y_test, test_pred)
    test_results.append(auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(min_samples_splits, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘min samples split’)
plt.show()

# min_samples_leaf
# min_samples_leaf is the minimum number of samples required to be at a leaf node.
# This parameter is similar to min_samples_splits, but this describe the number of 
# samples at the leafs.

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []

for each in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leafs=each, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    auc = roc_auc_score(y_train, train_pred)
    train_results.append(auc)
    
    test_pred = rf.predict(x_test)
    auc = roc_auc_score(y_test, test_pred)
    test_results.append(auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(min_samples_leafs, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘min samples leaf’)
plt.show()

# max_features
# This parameter represents the number of features to consider when splitting

max_features = list(range(1,train.shape[1]))
train_results = []
test_results = []

for each in max_features:
    rf = RandomForestClassifier(max_features=each, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    auc = roc_auc_score(y_train, train_pred)
    train_results.append(auc)
    
    test_pred = rf.predict(x_test)
    auc = roc_auc_score(y_test, test_pred)
    test_results.append(auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(max_features, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘max features’)
plt.show()
