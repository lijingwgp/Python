# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:42:28 2019

@author: Jing
"""

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
shap.initjs()

x,y = shap.datasets.adult()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
model = lgb.LGBMClassifier().fit(x_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)

#shap_values.shape
#x_test.shape
#shap_values[6]
#explainer.expected_value

#data_for_prediction = x_test.iloc[6]
#data_for_prediction_array = data_for_prediction.values.reshape(1,-1)
#model.predict_proba(data_for_prediction_array)

shap.force_plot(explainer.expected_value, shap_values[6,:], x_test.iloc[6,:])