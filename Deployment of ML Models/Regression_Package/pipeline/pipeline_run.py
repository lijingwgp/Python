# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:26:20 2020

@author: 607991
"""

import typing as t
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from Regression_Package.pipeline import pipeline
from Regression_Package.processing.data_management import (load_dataset, save_pipeline, load_pipeline)
from Regression_Package.config import config
from Regression_Package.processing.feature_validation import (validate_inputs)

###############################################################################

data = load_dataset(file_name=config.TRAINING_DATA_FILE)

X_train, X_test, y_train, y_test = train_test_split(data[config.FEATURES], 
                                                    data[config.TARGET], test_size=0.1, 
                                                    random_state=0)

y_train = np.log(y_train)
y_test = np.log(y_test)

pipeline.price_predict.fit(X_train[config.FEATURES], y_train)
save_pipeline(pipeline_to_persist=pipeline.price_predict)
trained_model = load_pipeline(file_name=config.TRAINED_MODEL_DIR)

X_train_validated = validate_inputs(input_data=X_train)
X_test_validated = validate_inputs(input_data=X_test)
prediction = np.log(trained_model.predict(X_test_validated[config.FEATURES]))
print(np.sqrt(metrics.mean_squared_error(y_test, prediction)))
