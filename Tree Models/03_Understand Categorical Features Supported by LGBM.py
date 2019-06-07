# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:36:05 2019

@author: jing.o.li
"""

# LightGBM allows one to specify directly categorical features and handles 
# those internally in a smart way, that might out-perform OHE. 
#
# Originally, I was puzzled about feature importance reported for such categorical features.
# After more research, it seems that:
#       1. the default implementation is not very useful, as there are several types of
#          importance values do not behave according to intuitive expectation;
#       2. it is beneficial to use SHAP package in python to produce stable feature
#          importance evaluation

# Now, let's look into how to use internal handling of categorical features in LGBM. 
# It turns out that the sklearn API of LGBM actually has those enabled by default. 
# In a sense that by default, it tries to guess which features are categorical, if you
# provided a pandas dataframe as input.

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# load data
application_train = pd.read_csv('application_train.csv')
y = application_train['TARGET']
x = application_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
del application_train

# Transform categorical features into the appropriate type tht is expected by LGBM
for each in x.columns:
    col_type = x[each].dtype
    if col_type == 'object' or col_type.name == 'categorical':
        x[each] = x[each].astype('category')
x.info()

# We will use LGBM classifier
# Train test split (80/20)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=314,stratify=y)

# Set parameters
fit_params={"early_stopping_rounds":10, 
            "eval_metric" : 'auc', 
            "eval_set" : [(x_test,y_test)],
            'eval_names': ['valid'],
            'verbose': 100,
            'feature_name': 'auto', # that's actually the default
            'categorical_feature': 'auto' # that's actually the default
            }

# Create model
clf = lgb.LGBMClassifier(num_leaves= 15, max_depth=-1, 
                         random_state=314, 
                         silent=True, 
                         n_jobs=-1, 
                         n_estimators=1000,
                         colsample_bytree=0.9,
                         subsample=0.9,
                         learning_rate=0.1)
clf.fit(x_train,y_train,**fit_params)

# Plot feature importance
feat_imp = pd.Series(clf.feature_importances_, index=x.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))
