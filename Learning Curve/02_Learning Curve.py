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
from statistics import mean 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from hyperopt.pyll.stochastic import sample
import random

file = open("Funnel_Final.pickle",'rb')
funnel = pickle.load(file)
file.close()

funnel = funnel.reset_index(drop=True)
x = funnel.drop(['LEAD_STATUS'],axis=1).values
y = funnel.LEAD_STATUS.values.reshape(-1,)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y)



################################
##### Baseline Performance #####
################################

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
1-roc_auc_score(y_test, y_pred)



########################################
##### Learning Curve Visualization #####
########################################

max_iters = 20
folds = 3
random_params = pd.DataFrame(columns = ['iteration','loss','params'], index = list(range(max_iters)))

param_grid = {
    'min_samples_split': list(range(2,20))
    #'min_samples_leaf': list(range(1, 10)), # arbitrarily set
    #'max_features': list(range(1,x_train.shape[1]))
    }

def RandomSearchCV_RF(iteration,params,folds):
    temp = []
    skf = StratifiedKFold(n_splits=folds)
    for train_index, val_index in skf.split(x_train, y_train):
        x_temp1, x_temp2 = x_train[train_index], x_train[val_index]
        y_temp1, y_temp2 = y_train[train_index], y_train[val_index]
        rf = RandomForestClassifier(n_jobs=-1,
                                    min_samples_split=params['min_samples_split'],
                                    #min_samples_leaf=params['min_samples_leaf'],
                                    #max_features=params['max_features']
                                    )
        rf.fit(x_temp1,y_temp1)
        y_temp = rf.predict(x_temp2)
        auc = roc_auc_score(y_temp2, y_temp)
        loss = 1-auc
        temp.append(loss)
    return [iteration,mean(temp),params]

for i in range(max_iters):
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    print(params)
    result_list = RandomSearchCV_RF(i,params,folds)
    random_params.loc[i,:] = result_list

random_params.sort_values('loss', ascending = True, inplace = True)

random_results = pd.DataFrame(columns = list(random_params.loc[0, 'params'].keys()),
                            index = list(range(len(random_params))))
for i, params in enumerate(random_params['params']):
    random_results.loc[i, :] = list(params.values())
for i, params in enumerate(random_params['loss']):
    random_results.loc[i, 'loss'] = params
for i, params in enumerate(random_params['iteration']):
    random_results.loc[i, 'iteration'] = params

sns.regplot('min_samples_split','loss', data = random_results)



######################################
##### Establishing Search Region #####
######################################

# min_samples_split sampling pool
min_samples_split = np.random.uniform(15,25,100)
sns.distplot(min_samples_split , hist=False, rug=True)
min_samples_split = [int(each) for each in min_samples_split]

param_grid = {
    'min_samples_split': min_samples_split
    #'min_samples_leaf': list(range(1, 10)), # arbitrarily set
    #'max_features': list(range(1,x_train.shape[1]))
    }

max_iters = 10
random_params = pd.DataFrame(columns = ['iteration','loss','params'], index = list(range(max_iters)))
for i in range(max_iters):
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    print(params)
    result_list = RandomSearchCV_RF(i,params,folds)
    random_params.loc[i,:] = result_list
random_params.sort_values('loss', ascending = True, inplace = True)

random_results = pd.DataFrame(columns = list(random_params.loc[0, 'params'].keys()),
                            index = list(range(len(random_params))))
for i, params in enumerate(random_params['params']):
    random_results.loc[i, :] = list(params.values())
for i, params in enumerate(random_params['loss']):
    random_results.loc[i, 'loss'] = params
for i, params in enumerate(random_params['iteration']):
    random_results.loc[i, 'iteration'] = params

sns.regplot('min_samples_split','loss', data = random_results)
random_params_best = random_params.loc[0, 'params'].copy()
