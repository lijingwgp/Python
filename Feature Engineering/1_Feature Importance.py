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
data['LEAD_STATUS'].replace('Stage 4 : Lead Closed Dismissed', 1, inplace=True)
data['LEAD_STATUS'].replace('Stage 5 : On Hold', 2, inplace=True)
data['LEAD_STATUS'].replace('Stage 6 : Converted', 3, inplace=True)
labels = data['LEAD_STATUS']
data = data.drop(['Unnamed: 0','LEAD_STATUS'], axis=1)

# =============================================================================
# # Option 1 - OHE
# features = pd.get_dummies(data)
# base_features = list(data.columns)
# one_hot_features = [column for column in features.columns if column not in base_features]
# 
# # Add one hot encoded data to original data
# data_all = pd.concat([features[one_hot_features], data], axis = 1)
# 
# # Extract feature names
# feature_names = list(data.columns)
# 
# # Convert to np array
# data = np.array(data)
# labels = np.array(labels).reshape((-1, ))
# 
# # Empty array for feature importances
# feature_importance_values = np.zeros(len(feature_names))
# 
# n_iterations = 10
# task = 'classification'
# early_stopping = True
# eval_metric = 'multi_logloss'
# 
# # Iterate through each fold
# for _ in range(n_iterations):
#     if task == 'classification':
#         model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1,
#                                    objective = 'multiclass', num_class = 3)
#     
#     elif task == 'regression':
#         model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)
# 
#     else:
#         raise ValueError('Task must be either "classification" or "regression"')
# 
#     # If training using early stopping need a validation set
#     if early_stopping:
#         train_features, valid_features, train_labels, valid_labels = train_test_split(data, labels, test_size = 0.15)
# 
#         # Train the model with early stopping
#         model.fit(train_features, train_labels, eval_metric = eval_metric,
#                   eval_set = [(valid_features, valid_labels)],
#                   early_stopping_rounds = 100, verbose = -1)
# 
#         # Clean up memory
#         gc.enable()
#         del train_features, train_labels, valid_features, valid_labels
#         gc.collect()
# 
#     else:
#         model.fit(data, labels)
# 
#     # Record the feature importances
#     feature_importance_values += model.feature_importances_ / n_iterations
# 
# feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
# 
# # Sort features according to importance
# feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
# 
# # Normalize the feature importances to add up to one
# feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
# feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
# 
# # Extract the features with zero importance
# record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
# to_drop = list(record_zero_importance['feature'])
# 
# # Make sure most important features are on top
# feature_importances = feature_importances.sort_values('cumulative_importance')
# 
# # Identify the features not needed to reach the cumulative_importance
# cumulative_importance = 0.95
# record_low_importance = feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]
# to_drop = list(record_low_importance['feature'])
# =============================================================================

# Option 2 - LGBM internal encoding
cat_cols = list(data.select_dtypes(include=['object']).columns)
col = cat_cols[0]
for col in cat_cols:
    # convert column to categorical
    data[col] = pd.Categorical(data[col])
    # extract categorical codes of these columns
    data[col] = data[col].cat.codes
    # convert the categorical codes to categorical
    data[col] = pd.Categorical(data[col])
# to remember which columns are categorical
cat_cols = [i for i,v in enumerate(data.dtypes) if str(v) == 'category']
x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=.2,stratify=labels)

# Set parameters
fit_params={
            "early_stopping_rounds": 10, 
            "eval_metric" : 'multi_logloss', 
            "eval_set" : [(x_test,y_test)],
            'verbose': 100,
            'feature_name': 'auto', # that's actually the default
            'categorical_feature': 'auto' # that's actually the default
            }

# Create model
clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, objective = 'multiclass', num_class= 3,)
clf.fit(x_train,y_train,**fit_params)

# Plot feature importance
feat_imp = pd.Series(clf.feature_importances_, index=data.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))
