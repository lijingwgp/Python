# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:37:06 2019

@author: jing.o.li
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:/Users/jing.o.li/Desktop/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == 'Yes')
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
x = data[feature_names]
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_x, train_y)

# We will look at SHAP values for a single row of the dataset (we arbitrarily chose row 5).
# For context, we'll look at the raw predictions before looking at the SHAP values.

# option1
data_for_prediction = val_x.iloc[0]
data_for_prediction_array = data_for_prediction.values.reshape(1,-1)
my_model.predict_proba(data_for_prediction_array)

# option2
data_for_prediction = val_x.iloc[:10]
data_for_prediction_array = data_for_prediction.values.reshape(10,-1)
my_model.predict_proba(data_for_prediction_array)

# The team is 70% likely to have a player win the award. Now we will move onto the 
# code to get SHAP values for that single prediction.

import shap
shap.initjs()

# create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate SHAP values
shap_values = explainer.shap_values(data_for_prediction)
shap_values

# The shap_values object above is a list with two arrays. The first array is the SHAP 
# values for a negative outcome (don't win the award), and the second array is the 
# list of SHAP values for the positive outcome (wins the award). We typically 
# think about predictions in terms of the prediction of a positive outsome, so we 
# will pull out Shap values for positive outcomes (shap_values[1])

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

# If we look carefully at the code where we are the SHAP values, we will notice that 
# they are created by the tree explainer from the SHAP package. There are also two
# other types of explainer, deep explainer and kernel explainer.
