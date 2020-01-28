# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:36:15 2020

@author: 607991
"""

import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from Regression_Package.config import config

###############################################################################

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return _data

def save_pipeline(*, pipeline_to_persist) -> None:
    save_path = config.TRAINED_MODEL_DIR
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR
    trained_model = joblib.load(filename=file_path)
    return trained_model
