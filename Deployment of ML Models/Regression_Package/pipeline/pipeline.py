# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:14:18 2020

@author: 607991
"""

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from Regression_Package.processing import features_transform as transf
from Regression_Package.processing import features_log as transf_log
from Regression_Package.config import config

###############################################################################

price_predict = Pipeline(
    [
        ('categorical_imputer',
            transf.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_inputer',
            transf.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),
        ('temporal_variable',
            transf.TemporalVariableEstimator(
                variables=config.TEMPORAL_VARS,
                reference_variable=config.DROP_FEATURES)),
        ('rare_label_encoder',
            transf.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=config.CATEGORICAL_VARS)),
        ('categorical_encoder',
            transf.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('log_transformer',
            transf_log.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),
        ('drop_features',
            transf.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('Linear_model', Lasso(alpha=0.005, random_state=0))
    ]
)
