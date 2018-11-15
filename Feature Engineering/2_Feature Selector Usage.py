# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:36:13 2018

@author: jing.o.li
"""

# In this notebook we will walk-through using the FeatureSelector class for selecting 
# features to remove from a dataset. This class has five methods for finding features
# to remove:
#       1. Find columns with a missing fraction greater than a specified threshold
#       2. Find features with only a single unique value
#       3. Find collinear features as identified by a correlation coefficient greater 
#          than a specified value
#       4. Find features with 0.0 importance from a gradient boosting machine
#       5. Find features that do not contribute to a specified cumulative feature 
#          importance from the gradient boosting machine




###################
### Preparation ###
###################

from FeatureEngineering import FeatureSelector
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.exceptions import NotFittedError
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError




######################
### Implementation ###
######################

train = pd.read_csv('application_train.csv')
train_labels = train['TARGET']
train = train.drop(columns = ['TARGET','SK_ID_CURR'])

# create the instance
fs = FeatureSelector(data = train, labels=train_labels)


### missing values
# The first feature selection method is straightforward: find any columns with a 
# missing fraction greater than a specified threshold. For this example we will use a 
# threhold of 0.6 which corresponds to finding features with more than 60% missing 
# values.

fs.identify_missing(missing_threshold=0.6)

# The features identified for removal can be accessed through the ops dictionary of the FeatureSelector object.
missing_features = fs.ops['missing']

# we can also plot a histogram of the missing column fraction for all columns in the dataset
fs.plot_missing() 

# for detailed information on the missing fractions
fs.missing_stats.head(10)


### single unique value
# The next method is straightforward: find any features that have only a single unique value.

fs.identify_single_unique()
single_unique = fs.ops['single_unique']
single_unique
fs.plot_unique()
fs.unique_stats.sample(5)


### collinear features
# This method finds pairs of collinear features based on the Pearson correlation 
# coefficient. For each pair above the specified threshold (in terms of absolute value), 
# it identifies one of the variables to be removed. We need to pass in a correlation_threshold.
#
# For each pair, the feature that will be removed is the one that comes last in terms of the 
# column ordering in the dataframe.

fs.identify_collinear(correlation_threshold=.97)
correlated_features = fs.ops['collinear']
correlated_features[:5]
fs.plot_collinear()
fs.plot_collinear(plot_all=True)
fs.record_collinear.head()


### zero importance features
# This method relies on a machine learning model to identify features to remove. 
# It therefore requires a supervised learning problem with labels. The method works by
# finding feature importances using a gradient boosting machine implemented in the LightGBM library.
#
#       1. task: either classification or regression
#       2. eval_metric: the metric used for early stopping
#       3. n_iterations: number of training runs. the feature importantce are averaged over training runs
#       4. early_stopping: stops training when the performance on a validation set no longer decreases

fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)

# Running the gradient boosting model requires one hot encoding the features. 
# These features are saved in the one_hot_features attribute.
# The original features are saved in the base_features.

one_hot_features = fs.one_hot_features
base_features = fs.base_features
#train.dtypes
#train.select_dtypes(include=['object'])
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))

# After one-hot encoding, the data_all attribute holds the original data plus the one-hot encoded features.
fs.data_all.head(10)

# there are a number of methods we can use to inspect the results of the reature importance
zero_importance_features = fs.ops['zero_importance']
zero_importance_features[10:15]

# plot feature importances
#       The feature importance plot using plot_feature_importances will show us the 
#       plot_n most important features.
#       It also shows us the cumulative feature importance versus the number of features.

fs.plot_feature_importances()
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)

# All of the feature importances are accessible in the feature_importances attribute 
fs.feature_importances.head(10)

# If we want the top 100 most importance, we could do the following.
one_hundred_features = list(fs.feature_importances.loc[:99, 'feature'])
len(one_hundred_features)


### low importance features
# This method builds off the feature importances from the gradient boosting machine 
# (identify_zero_importance must be run first) by finding the lowest importance 
# features not needed to reach a specified cumulative total feature importance.
#
# If we pass in 0.99, this will find the lowest important features that are not needed
# to reach 99% of the total feature importance.
#
# When using this method, we must have already run identify_zero_importance and need 
# to pass in a cumulative_importance that accounts for that fraction of total feature 
# importance.

fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']
low_importance_features[:5]


### alternative option for using all methods
# If we don't want to run the identification methods one at a time, we can use 
# identify_all to run all the methods in one call. For this function, we need to
# pass in a dictionary of parameters to use for each individual identification method.

fs = FeatureSelector(data = train, labels = train_labels)
fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})
list(fs.ops)




#########################
### Removing Features ###
#########################

# Once we have identified the features to remove, we have a number of ways to drop
# the features. We can access any of the feature lists in the removal_ops dictionary
# and remove the columns manually. We also can use the remove method, passing in the 
# methods that identified the features we want to remove. 
#
# It is often a good idea to inspect the features that will be removed before using
# the remove function.

# To remove the features from all of the methods, pass in method='all'. Before we do 
# this, we can check how many features will be removed using check_removal. This 
# returns a list of all the features that have been idenfitied for removal.

all_to_remove = fs.check_removal()
all_to_remove[10:25]

# now we can remove all of the features identified
train_removed = fs.remove(methods = 'all')
# or
train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])

### handling one-hot features
# If we look at the dataframe that is returned, we may notice several new columns 
# that were not in the original data. These are created when the data is one-hot 
# encoded for machine learning.
#
# To remove all the one-hot features, we can pass in keep_one_hot = False to the 
# remove method.

train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)
print('Original Number of Features', train.shape[1])
print('Final Number of Features: ', train_removed_all.shape[1])

