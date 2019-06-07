# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:03:16 2019

@author: jing.o.li
"""

import pandas
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
from paramsearch import paramsearch
from itertools import product,chain

# read in the train and test data from csv files
colnames = ['age','wc','fnlwgt','ed','ednum','ms','occ','rel','race','sex','cgain','closs','hpw','nc','label']
train_set = pandas.read_csv("adult.data.txt",header=None,names=colnames,na_values='?')
test_set = pandas.read_csv("adult.test.txt",header=None,names=colnames,na_values='?',skiprows=[0])

# convert categorical columns to integers
category_cols = ['wc','ed','ms','occ','rel','race','sex','nc']
cat_dims = [train_set.columns.get_loc(i) for i in category_cols] 
for header in category_cols:
    train_set[header] = train_set[header].astype('category').cat.codes
    test_set[header] = test_set[header].astype('category').cat.codes

# split labels out of data sets    
train_label = train_set['label']
train_set = train_set.drop('label', axis=1)
test_label = test_set['label']
test_set = test_set.drop('label', axis=1)

# parameter domian
params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200],
          'ctr_border_count':[50,5,10,20,100,200],
          'thread_count':4}

# this function does 3-fold crossvalidation with catboostclassifier          
def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):
    kf = KFold(n_splits=n_splits,shuffle=True) 
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index,:]
        test = train_set.iloc[test_index,:]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = cb.CatBoostClassifier(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))
    return np.mean(res)

# this function runs grid search on several parameters
def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):
    ps = paramsearch(params)
    # search 'border_count', 'l2_leaf_reg' etc. individually 
    #   but 'iterations','learning_rate' together
    for prms in chain(ps.grid_search(['border_count']),
                      ps.grid_search(['ctr_border_count']),
                      ps.grid_search(['l2_leaf_reg']),
                      ps.grid_search(['iterations','learning_rate']),
                      ps.grid_search(['depth'])):
        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
        # save the crossvalidation result so that future iterations can reuse the best parameters
        ps.register_result(res,prms)
        print(res,prms,s'best:',ps.bestscore(),ps.bestparam())
    return ps.bestparam()

bestparams = catboost_param_tune(params,train_set,train_label,cat_dims)

# train classifier with tuned parameters    
clf = cb.CatBoostClassifier(**bestparams)
clf.fit(train_set, np.ravel(train_label), cat_features=cat_dims)
res = clf.predict(test_set)
print('error:',1-np.mean(res==np.ravel(test_label)))








