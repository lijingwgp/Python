# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:14:14 2018

@author: jing.o.li
"""

# In this notebook, we will walk through automated hyperparameter tuning using Bayesian
# Optimization. Specifically, we will optimize the hyperparameters of a Gradient Boosting
# model using the Hyperopt library (with the TPE algorithm). We will compare the results
# of random search for hyperparameter tuning with the Bayesian model-based optimization
# method



##################
#### Hyperopt ####
##################

# Hyperopt is one of several automated hyperparameter tuning libraries using Bayesian
# optimization. These libraries differ in the algorithm used to both construct the surrogate
# of the objective function and choose the next hyperparameters to evaluate in the objective
# function. Hyperopt uses the Tree Parzen Estimator (TPE). Other Python libararies include
# Spearmint, which uses a Gaussian process for the surrogate, and SMAC, which uses a random
# forest regression. 

# Hyperopt ha a simple syntax for structuring an optimization problem which extends beyond
# hyperparameter tuning to any problem that involves minimizing a function. Moreover, the 
# structure of a Bayesian Optimization problem is similar across the libraries, with the 
# major differences coming in the syntax.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

max_iters = 500
folds = 5



##############
#### Data ####
##############

# In this data set, the objective is to determine whether or not a potential customer
# will buy an insurance policy by training a model on past data. 

data = pd.read_csv('caravan-insurance-challenge.csv')
train = data[data['ORIGIN'] == 'train']
test = data[data['ORIGIN'] == 'test']

# Extract the labels and format properly
train_labels = np.array(train['CARAVAN'].astype(np.int32)).reshape((-1,))
test_labels = np.array(test['CARAVAN'].astype(np.int32)).reshape((-1,))

# Drop the unneeded columns
train = train.drop(columns = ['ORIGIN', 'CARAVAN'])
test = test.drop(columns = ['ORIGIN', 'CARAVAN'])

# Convert to numpy array for splitting in cross validation
train_features = np.array(train)
test_features = np.array(test)
train_labels = train_labels[:]

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
train.head()

# We can inspect the class ratio
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.hist(train_labels, edgecolor = 'k'); 
plt.xlabel('Label'); plt.ylabel('Count'); plt.title('Counts of Labels');

# This is an imbalanced class problem: there are far more observations where an 
# insurance policy was not bought (0) than when the policy was bought (1).
#
# Therefore, accuracy is a poor metric to use for this task. Instead, we will use the 
# common classification metric of Receiver Operating Characteristic Area Under the 
# Curve (ROC AUC).

# We will use the LightGBM implementation of the gradient boosting machine. 
# This is much faster than and achieves results comparable to XGBoost.

model = lgb.LGBMClassifier()

# All we need to do is fit the model on the training data and make predictions on the 
# testing data. For the predictions, because we are measuring ROC AUC and not accuracy, 
# we have the model predict probabilities and not hard binary values.

from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

start = timer()
model.fit(train_features, train_labels)
train_time = timer() - start

predictions = model.predict_proba(test_features)[:, 1]
auc = roc_auc_score(test_labels, predictions)

print('The baseline score on the test set is {:.4f}.'.format(auc))
print('The baseline training time is {:.4f} seconds'.format(train_time))

# That's our metric to beat. Due to the small size of the dataset.



###################################################
#### Random Sampling in the Random Search Grid ####
###################################################

# Random search and Bayesian optimization both search for hyperparameters from a domain.
# For random search this domain is called a hyperparameter grid and uses discrete 
# values for the hyperparameters.
#
# First, let's took a look at the hyperparameters that need to be tuned.
lgb.LGBMClassifier()

# Base of off these default values, we can build a hyperparameter grid. 
param_grid = {
    'boosting_type': ['gbdt'],
    'num_leaves': list(range(30, 80)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10))
}

# Let's look at two of the distributions, the learning_rate and the num_leaves. 
# The learning rate is typically represented by a logarithmic distribution because it can vary over several orders of magnitude. np.logspace returns values evenly spaced over a log-scale

plt.hist(param_grid['learning_rate'], color = 'r', edgecolor = 'k');
plt.xlabel('Learning Rate', size = 14); 
plt.ylabel('Count', size = 14); plt.title('Learning Rate Distribution', size = 18);

plt.hist(param_grid['num_leaves'], color = 'm', edgecolor = 'k')
plt.xlabel('Learning Number of Leaves', size = 14); plt.ylabel('Count', size = 14); 
plt.title('Number of Leaves Distribution', size = 18);

# Now, lets take a look at how we randomly sample a set of hyperparameters from our 
# grid using a dictionary comprehension.

import random
params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
params



##########################################################
#### Cross Validation with Early Stopping in LightGBM ####
##########################################################

# The scikit-learn cross validation api does not include the option for early stopping.
# Therefore, we will use the LightGBM cross validation function with 100 early stopping
# rounds. To use this function, we need to create a dataset from our features and labels.
#
# The cv function takes in the parameters, the training data, the number of training 
# rounds, the number of folds, the metric, the number of early stopping rounds, and 
# a few other arguments.

train_set = lgb.Dataset(train_features, label = train_labels)

# We set the number of boosting rounds very high, but we will not actually train 
# this many estimators because we are using early stopping to stop training when the 
# validation score has not improved for 100 estimators.

# Perform cross validation with 10 folds
try1 = lgb.cv(params, train_set, num_boost_round = 10000, nfold = 5, metrics = 'auc', 
           early_stopping_rounds = 100, verbose_eval = False, seed = 50)

# Highest score
try1_best = np.max(try1['auc-mean'])
# Standard deviation of best score
try1_best_std = try1['auc-stdv'][np.argmax(try1['auc-mean'])]

print('The maximium ROC AUC on the validation set was {:.5f} with std of {:.5f}.'.format(try1_best, try1_best_std))
print('The ideal number of iterations was {}.'.format(np.argmax(try1['auc-mean']) + 1))



##################################################
#### Results Dataframe and Objective Function ####
##################################################

# We have our domain and our algorithm which in this case is random selection. 
# The other two parts we need for an optimization problem are an objective function 
# and a data structure to keep track of the results

random_results = pd.DataFrame(columns = ['loss', 'params', 'iteration', 'estimators', 'time'],
                       index = list(range(max_iters)))

# The objective function will take in the hyperparameters and return the validation loss.
# We already choose our metric as ROC AUC and now we need to figure out how to measure 
# it. We can't evaluate the ROC AUC on the test set because that would be cheating. 
# Instead we must use a validation set to tune the model and hope that the results 
# translate to the test set.
#
# A better approach than drawing the validation set from the training data is KFold cross validation. 
# In addition to not limiting the training data, this method should also give us a 
# better estimate of generalization error on the test set because we will be using 
# K validations rather than only one. 
#
# In the case of random search, the next values selected are not based on the past 
# evaluation results, but we clearly should keep track so we know what values worked 
# the best

def random_objective(params, iteration, n_folds = folds):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""

    # Perform n_folds cross validation
    start = timer()
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    end = timer()
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)
    
    # Return list of results
    return [loss, params, iteration, n_estimators, end - start]



######################################
#### Random Search Implementation ####
######################################

random.seed(50)

# Iterate through the specified number of evaluations
for i in range(max_iters):
    
    # randomly sampling
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    print(params)
    
    # lightgbm cv
    results_list = random_objective(params, i)
    
    # add results to next row in dataframe
    random_results.loc[i, :] = results_list

random_results.sort_values('loss', ascending = True, inplace = True)
random_results.reset_index(inplace = True, drop = True)
random_results.head()

# Random Search Performance
# Recall that the baseline gradient boosting model achieved a score of 0.71 on the 
# training set. We can use the best parameters from random search and evaluate them 
# on the testing set.

# Find the best parameters and number of estimators. The estimators key holds the 
# average number of boosting trained with early stopping. We can use this as the 
# optimal number of boosting iterations in the gradient boosting model.

best_random_params = random_results.loc[0, 'params'].copy()
best_random_estimators = int(random_results.loc[0, 'estimators'])
best_random_model = lgb.LGBMClassifier(n_estimators=best_random_estimators, n_jobs = -1, 
                                       objective = 'binary', **best_random_params, random_state = 50)
best_random_model.fit(test_features, test_labels)
predictions = best_random_model.predict_proba(test_features)[:, 1]

print('The best model from random search scores {:.4f} on the test data.'.format(roc_auc_score(test_labels, predictions)))
print('This was achieved using {} search iterations.'.format(random_results.loc[0, 'iteration']))
