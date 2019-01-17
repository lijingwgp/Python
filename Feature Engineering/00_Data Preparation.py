# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 13:46:43 2018

@author: GJC8C0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import KFold
from sklearn.exceptions import NotFittedError
from itertools import chain

drop_list = ['UBK_PHONE_EXT','HOUSEHOLD_HIGH_INCOME','UBK_TYPE__C','ACCOUNT_URBANIZATION_CODE__C','ACCOUNT_FUNCTION_STATUS_MSG__C',
'FED_TAX_ID','FED_TAX_ID_TYPE','SALES_CODE_CORP','SBE_BUSINESS_TYPE_CODE','WBENC_IND','GAY_LESBIAN_IND','NMSDC_IND',
'DISABLED_IND','SBE_8A','HBCU_IND','SDB_IND','CA_CALTRANS_IND','HUBZONE_IND','MWBE_IND','SBE_BUSINESS_LOCATION_CODE',
'PRIM_REGION','CA_PUC_IND','FED_ENTITY_IND','DISABLED_VET_SB_IND','COUNT_SR_STANDARD','DISABLED_VET_IND','DISADVANTAGE_BUS_IND',
'COUNT_SR_PERIODICALS','FOREIGN_IND','COUNT_SR_EXPRESS','VET_SBE_IND','VET_IND','HOURS_CNT_24_TO_72','LESS_7_DAYS_CNT',
'STOCK_EX_CODE','TICKER_SYM','DAYS_CNT_61_90','LESS_24_HOURS_CNT','LESS_30_DAYS_CNT','MBE_IND','DAYS_CNT_30_60','COUNT_SR_INTERNATIONAL',
'OOB_IND','SBE_LIABILITY_INDICATOR','GOV_ENTITY_IND','COUNT_SR_PACKAGE','CIO_NAME','CIO_TITLE_DESC','COUNT_SR_PRIORITY','NONPROFIT_IND',
'COUNT_SR_FIRSTCLASS','CASES_OPEN','ACCOUNT_FAX','LEGAL_GUL_PARENT_IND','CASES_CLOSED','COUNT_SR_OTHER','CFO_NAME','CFO_TITLE_DESC',
'WSBE_IND','PUBLIC_IND','DAYS_CNT_90DAYS','WBE_IND','ACCOUNTID','NUM_CASES_ACCT','BSN_PROACTIVE_FREQUENCY__C','LEGAL_PARENT_IND',
'SBE_ANNUAL_SALES_MAX','SBE_ANNUAL_SALES_MIN','MWBE_STATUS','SBE_NUMBER_OF_EMPLOYEE_MAX','SBE_NUMBER_OF_EMPLOYEE_MIN',
'LBE_IND','NONSBE_IND','PARENT_STATE','PARENT_ZIP','PARENT_ADDRESS','PARENT_CITY','PARENT_CTRY_ISOCD','PARENT_NAME',
'ACCT_BSN_REP_DISTRICT__C','BSN_REP_DISTRICT__C','UBK_ADDRESS2','ACCT_BSN_REP_AREA__C','BSN_REP__C','BSN_REP_AREA__C',
'BSN_ROLE_ID_FORMULA__C','ACCOUNT_ANNUALREVENUE','GLOBAL_ULTIMATE_STATE','SEC_ZIP4','SEC_STATE','SEC_ADDRESS','SEC_CITY',
'SEC_ZIPCODE','DOM_ULTIMATE_ADDRESS','DOM_ULTIMATE_CITY','DOM_ULTIMATE_CTRY_ISOCD','DOM_ULTIMATE_NAME','DOM_ULTIMATE_STATE',
'DOM_ULTIMATE_ZIP','GLOBAL_ULTIMATE_ZIP','GLOBAL_ULTIMATE_CITY','GLOBAL_ULTIMATE_ADDRESS','GLOBAL_ULTIMATE_CTRY_ISOCD',
'GLOBAL_ULTIMATE_NAME','LEGAL_LINKED_RECORD_IND','OPTY_APPROVAL_STATUS','sum_totalprice','SALE_VALUE','CEO_NAME','CEO_TITLE_DESC',
'UBK_DBA_NAME','ACCOUNT_STATUS__C','GUEST_ACCOUNT_USER_ID__C','PMSA_CODE','BUS_LEGAL_CODE','BUS_LEGAL_STATUS','BUS_DESC',
'UBK_CTRY','EMAIL_ADDRESS','PRE_PRODUCT_2','ACCOUNT_FUNCTION_STATUS__C','LEGAL_NAME','CONVERTED_OPTY_ID','OPTY_STAGENAME',
'FAXPHONE_NUMBER','INDIV_FIRM_CODE','ACCOUNT_PHONE','WEB_ADDRESS','CMSA_DESC','TITLE_DESC','TITLE_CODE','ZIP_CODE_INDEX__C',
'EMPLOYEE_CODE_SITE','LAST_NAME','FIRST_NAME','PRIM_ZIP4','PRIM_SIC_DESC','CONGRESS','ETHNICITY','GENDER','SHIPPINGPOSTALCODE',
'SHIPPINGSTREET','PHONE_NUMBER','BUS_DESC_IND','CEO_NAME_IND','CFO_NAME_IND','CIO_NAME_IND','UBK_ZIP4','BUSINESS_NAME',
'PRIM_ADDRESS','PRIM_ZIPCODE','UBK_PHONE','BILLING_ZIP_CODE_EXTENSION__C','BILLINGPOSTALCODE','BILLINGSTREET','GEO_ZIPCODE',
'DISTRICT_DESC','AREA_DESC','UBK_ADDRESS','UBK_BUSINESS_NAME','UBK_ZIPCODE','count','EFXID','ACCOUNT_ID','DISTRICT_ID',
'AREA_ID','UBK']

hist_data = pd.read_csv('LR_EDA_121818.csv', low_memory=False)
hist_data1 = [each for each in hist_data.columns if each not in drop_list]
hist_data1 = hist_data[hist_data1]



##########################################
### Feature Engineering -- Aggregation ###
##########################################

temp1 = ['Q1','Q2','Q3','Q4']
to_aggregate = [each for each in hist_data1 if ((each[-2:] in temp1))]
quarterly_revenue = hist_data1[to_aggregate]
quarterly_revenue = quarterly_revenue.fillna(0)
quarterly_revenue_names = list(quarterly_revenue)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
for group in chunker(quarterly_revenue_names, 4):    
    temp2 = quarterly_revenue[group].columns.str.split('_').str[0].tolist()
    temp2 = temp2[0]
    quarterly_revenue[temp2] = quarterly_revenue[group].sum(axis=1)
quarterly_revenue.to_csv('revenue_vars.csv', sep=',')

hist_data2 = [each for each in hist_data1.columns if ((each[-2:] not in temp1))]
hist_data2 = hist_data1[hist_data2]
quarterly_revenue_names = [each for each in quarterly_revenue.columns if ((each[-2:] not in temp1))]
quarterly_revenue = quarterly_revenue[quarterly_revenue_names]
hist_data2 = pd.concat([hist_data2, quarterly_revenue], axis=1)
del hist_data1



######################
### Missing fields ###
######################

# Calculate the fraction of missing in each column
missing_series = hist_data2.isnull().sum() / hist_data2.shape[0]
missing_series = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})
missing_series = missing_series.sort_values('missing_fraction', ascending = False)

# Find the columns with a missing percentage above the threshold
record_missing = pd.DataFrame(missing_series.loc[missing_series['missing_fraction'] > .7]).reset_index().rename(columns = {'index':'feature',0:'missing_fraction'})
to_drop_missing = list(record_missing['feature'])

hist_data3 = [each for each in hist_data2.columns if each not in to_drop_missing]
hist_data3 = hist_data2[hist_data3]
del hist_data2



###########################
### Single Unique Value ###
###########################

# Calculate the unique counts in each column
unique_counts = hist_data3.nunique()
unique_counts = pd.DataFrame(unique_counts).rename(columns = {'index': 'feature', 0: 'unique'})
unique_counts = unique_counts.sort_values('unique', ascending = True)

# Find the columns with only one unique count
record_single = pd.DataFrame(unique_counts.loc[unique_counts['unique'] == 1]).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
to_drop_single = list(record_single['feature'])

hist_data4 = [each for each in hist_data3.columns if each not in to_drop_single]
hist_data4 = hist_data3[hist_data4]
del hist_data3



##########################
### Categorical Levels ###
##########################

# Separate out the categorical variables from the numerical variables
first_look = hist_data4.columns.to_series().groupby(hist_data4.dtypes).groups

# Convert all string columns to categorical columns then count their levels
df_str = hist_data4.loc[:, hist_data4.dtypes == np.object]
df_str = df_str.astype('category')
categorical_levels = df_str.nunique()
categorical_levels = pd.DataFrame(categorical_levels).rename(columns = {'index': 'feature', 0: 'levels'})
categorical_levels = categorical_levels.sort_values('levels', ascending = False)

# Delete columns that have too many levels
record_levels = pd.DataFrame(categorical_levels.loc[categorical_levels['levels'] > 60]).reset_index().rename(columns = {'index': 'feature', 0: 'levels'})
to_drop_levels = list(record_levels['feature'])

hist_data5 = [each for each in hist_data4.columns if each not in to_drop_levels]
hist_data5 = hist_data4[hist_data5]
del hist_data4



##########################
### Collinear Features ###
##########################

# All numeric columns
df_num1 = hist_data5.loc[:, hist_data5.dtypes == np.int64]
df_num2 = hist_data5.loc[:, hist_data5.dtypes == np.float64]
df_num = pd.concat([df_num1, df_num2], axis = 1)
df_num = df_num.fillna(0)

# Calculate correlation matrix and extract the upper triangle of the correlation matrix
corr_matrix = df_num.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Select the features with correlations above the threshold
# Need to use the absolute value
to_drop_collinear = [column for column in upper.columns if any(upper[column].abs() > .6)]

# Dataframe to hold correlated pairs
record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

# Iterate through the columns to drop to record pairs of correlated features
for column in to_drop_collinear:
    # Find the correlated features
    corr_features = list(upper.index[upper[column].abs() > .6])
    # Find the correlated values
    corr_values = list(upper[column][upper[column].abs() > .6])
    drop_features = [column for _ in range(len(corr_features))]    
    # Record the information (need a temp df for now)
    temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                      'corr_feature': corr_features,
                                      'corr_value': corr_values})
    # Add to dataframe
    record_collinear = record_collinear.append(temp_df, ignore_index = True)

to_drop_collinear_final = ['BUSINESS_PHONE_COUNT','PHONE_EXT_COUNT','NAME_COUNT','EMAIL_COUNT',
                           'EFX_FIRST_NAME_IND','EFX_LAST_NAME_IND','PARENT_NAME_IND','INDUSTRY2',
                           'PRIM_CTRY_NUM']

hist_data6 = [each for each in hist_data5.columns if each not in to_drop_collinear_final]
hist_data6 = hist_data5[hist_data6]
del hist_data5



######################
### Sparse Columns ###
######################

feature_a = list(hist_data6.select_dtypes(include=['int64']))
feature_b = list(hist_data6.select_dtypes(include=['float64']))
feature_a_result = hist_data6[feature_a].agg(['sum'])
feature_b_result = hist_data6[feature_b].agg(['sum'])

sparse1 = [each for each in feature_a_result.columns if (abs(feature_a_result[each]) < 100).any()]
sparse2 = [each for each in feature_b_result.columns if (abs(feature_b_result[each]) < 100).any()]
sparse = sparse1 + sparse2

hist_data7 = [each for each in hist_data6.columns if each not in sparse]
hist_data7 = hist_data6[hist_data7]
del hist_data6
hist_data7.to_csv('prepared.csv')
