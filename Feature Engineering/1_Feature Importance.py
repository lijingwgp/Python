# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:06:55 2019

@author: jing.o.li
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split

# Import data
data = pd.read_csv('prepared.csv', low_memory=False)
labels = data['LEAD_STATUS']
data = data.drop(['Unnamed: 0','LEAD_STATUS'], axis=1)

# =============================================================================
# # One hot encoding
# features = pd.get_dummies(data)
# base_features = list(data.columns)
# one_hot_features = [column for column in features.columns if column not in base_features]
# 
# # Add one hot encoded data to original data
# data_all = pd.concat([features[one_hot_features], data], axis = 1)
# =============================================================================

# Extract feature names
feature_names = list(data.columns)

# Convert to np array
data = np.array(data)
labels = np.array(labels).reshape((-1, ))

# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))

n_iterations = 10
task = 'classification'
early_stopping = True
eval_metric = 'auc'

# Iterate through each fold
for _ in range(n_iterations):
    if task == 'classification':
        model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
    
    elif task == 'regression':
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)
    
    else:
        raise ValueError('Task must be either "classification" or "regression"')
                
    # If training using early stopping need a validation set
    if early_stopping:
        train_features, valid_features, train_labels, valid_labels = train_test_split(data, labels, test_size = 0.15)

        # Train the model with early stopping
        model.fit(train_features, train_labels, eval_metric = eval_metric,
                  eval_set = [(valid_features, valid_labels)],
                  early_stopping_rounds = 100, verbose = -1)
        
        # Clean up memory
        gc.enable()
        del train_features, train_labels, valid_features, valid_labels
        gc.collect()
                
    else:
        model.fit(data, labels)

    # Record the feature importances
    feature_importance_values += model.feature_importances_ / n_iterations

feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
# Sort features according to importance
feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

# Normalize the feature importances to add up to one
feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

# Extract the features with zero importance
record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
to_drop = list(record_zero_importance['feature'])

# Make sure most important features are on top
feature_importances = feature_importances.sort_values('cumulative_importance')

# Identify the features not needed to reach the cumulative_importance
cumulative_importance = 0.95
record_low_importance = feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]
to_drop = list(record_low_importance['feature'])

