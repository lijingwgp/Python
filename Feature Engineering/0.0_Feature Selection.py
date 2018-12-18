# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:46:43 2018

@author: GJC8C0
"""

import pandas as pd
import numpy as np
#import lightgbm as lgb
from sklearn.model_selection import KFold
#import matplotlib.pyplot as plt
#import seaborn as sns
import gc
from sklearn.exceptions import NotFittedError
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

# Load data
hist_data = pd.read_csv('LR_PRE_EDA1_120718.csv', low_memory=False)
var_list = pd.read_csv('Var_Divide_12_11.csv', low_memory=False)
var_list1 = list(var_list.Name)
hist_data1 = hist_data[var_list1]



######################
### Missing fields ###
######################

# Calculate the fraction of missing in each column
missing_series = hist_data1.isnull().sum() / hist_data1.shape[0]
missing_series = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
missing_series = missing_series.sort_values('missing_fraction', ascending = False)

# Find the columns with a missing percentage above the threshold
record_missing = pd.DataFrame(missing_series.loc[missing_series['missing_fraction'] > .7]).reset_index().rename(columns = {'index':'feature',0:'missing_fraction'})
to_drop_missing = list(record_missing['feature'])



###########################
### Single Unique Value ###
###########################

# Calculate the unique counts in each column
unique_counts = hist_data1.nunique()
unique_counts = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'unique'})
unique_counts = unique_counts.sort_values('unique', ascending = True)

# Find the columns with only one unique count
record_single = pd.DataFrame(unique_counts.loc[unique_counts['unique'] == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
to_drop_single = list(record_single['feature'])



##########################
### Categorical Levels ###
##########################

# Separate out the categorical variables from the numerical variables
first_look = hist_data1.columns.to_series().groupby(hist_data1.dtypes).groups

# Convert all string columns to categorical columns then count their levels
df_str = hist_data1.loc[:, hist_data1.dtypes == np.object]
df_str = df_str.astype('category')
categorical_levels = df_str.nunique()
categorical_levels = pd.DataFrame(categorical_levels).rename(columns = {'index': 'feature', 0: 'levels'})
categorical_levels = categorical_levels.sort_values('levels', ascending = False)

# Delete columns that have too many levels
record_levels = pd.DataFrame(categorical_levels.loc[categorical_levels['levels'] > 60]).reset_index().rename(columns = {'index': 'feature', 0: 'levels'})
to_drop_levels = list(record_levels['feature'])



##########################
### Collinear Features ###
##########################

# All numeric columns
df_num1 = hist_data1.loc[:, hist_data1.dtypes == np.int64]
df_num2 = hist_data1.loc[:, hist_data1.dtypes == np.float64]
df_num = pd.concat([df_num1, df_num2], axis = 1)
to_impute = list(df_num)

# Calculate correlation matrix and extract the upper triangle of the correlation matrix
corr_matrix = df_num.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Select the features with correlations above the threshold
# Need to use the absolute value
to_drop_collinear = [column for column in upper.columns if any(upper[column].abs() > .5)]

# Dataframe to hold correlated pairs
record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

# Iterate through the columns to drop to record pairs of correlated features
for column in to_drop_collinear:
    # Find the correlated features
    corr_features = list(upper.index[upper[column].abs() > .5])
    # Find the correlated values
    corr_values = list(upper[column][upper[column].abs() > .5])
    drop_features = [column for _ in range(len(corr_features))]    
    # Record the information (need a temp df for now)
    temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                      'corr_feature': corr_features,
                                      'corr_value': corr_values})
    # Add to dataframe
    record_collinear = record_collinear.append(temp_df, ignore_index = True)
