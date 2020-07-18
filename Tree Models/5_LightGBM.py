# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:14:01 2019

@author: Jing
"""

############################
##### Data Preparation #####
############################

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

from plotnine import *
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV

train = './adult-training.csv'
test = './adult-test.csv'
columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus','Occupation',
           'Relationship','Race','Sex','CapitalGain','CapitalLoss','HoursPerWeek',
           'Country','Income']

df_train = pd.read_csv(train, names=columns)
df_test = pd.read_csv(test, names=columns, skiprows=1)
df_train.drop('fnlgwt', axis=1, inplace=True)
df_test.drop('fnlgwt', axis=1, inplace=True)
df_train.head()



##########################
##### Data Cleansing #####
##########################

# first, we need to remove any spcial characters from all columns. 
# second, any space or "." characters too will be removed from any str data.

for col in df_train.columns:
    df_train[col].replace(' ?', 'Unknown', inplace=True)
    df_test[col].replace(' ?', 'Unknown', inplace=True)

for col in df_train.columns:
    if df_train[col].dtype != 'int64':
        df_train[col] = df_train[col].apply(lambda x: x.replace(" ", ""))
        df_train[col] = df_train[col].apply(lambda x: x.replace(".", ""))
        df_test[col] = df_test[col].apply(lambda x: x.replace(" ", ""))
        df_test[col] = df_test[col].apply(lambda x: x.replace(".", ""))

# there are two columns that describe education of individuals, I would assume both
# of these to be highly correlated and hence remove the Education column.
#
# the country column too should not play a role in prediction of income and hence
# we remove that as well

df_train.drop(['Country','Education'], axis=1, inplace=True)
df_test.drop(['Country','Education'], axis=1, inplace=True)

# although Age and EdNum columns are numeric, they can be easily binned and be more effective. 
# we will bine age in bins of 10 and number of years of education into bins of 5

colnames = list(df_train)
colnames.remove('Age')
colnames.remove('EdNum')   
colnames = ['AgeGroup', 'Education'] + colnames

labels = ["{0}-{1}".format(i, i+9) for i in range(0,100,10)]
df_train['AgeGroup'] = pd.cut(df_train.Age, range(0,101,10), right=False, labels=labels)
df_test['AgeGroup'] = pd.cut(df_test.Age, range(0,101,10), right=False, labels=labels)
labels = ["{0}-{1}".format(i, i+4) for i in range(0,20,5)]
df_train['Education'] = pd.cut(df_train.EdNum, range(0, 21, 5), right=False, labels=labels)
df_test['Education'] = pd.cut(df_test.EdNum, range(0, 21, 5), right=False, labels=labels)

df_train = df_train[colnames]
df_test = df_test[colnames]

# now that we have cleaned the data, let's look how balanced out data set is
df_train.Income.value_counts()
df_test.Income.value_counts()

# note that this is a very imbalanced dataset, we should do resampling
# but for simplicity, we will treat this dataset as a regular problem.

mapper = DataFrameMapper([
    ('AgeGroup', LabelEncoder()),
    ('Education', LabelEncoder()),
    ('Workclass', LabelEncoder()),
    ('MaritalStatus', LabelEncoder()),
    ('Occupation', LabelEncoder()),
    ('Relationship', LabelEncoder()),
    ('Race', LabelEncoder()),
    ('Sex', LabelEncoder()),
    ('Income', LabelEncoder())
], df_out=True, default=None)

cols = list(df_train.columns)
cols.remove("Income")
cols = cols[:-3] + ["Income"] + cols[-3:]

df_train = mapper.fit_transform(df_train.copy())
df_train.columns = cols

df_test = mapper.transform(df_test.copy())
df_test.columns = cols

cols.remove("Income")
x_train, y_train = df_train[cols].values, df_train["Income"].values
x_test, y_test = df_test[cols].values, df_test["Income"].values



#################################
##### Plot Confusion Matrix #####
#################################

# Using the following code we can plot the confusion matrix for any of the 
# tree-based models.

def plot_confusion_matrix(cm, classes, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    cmap = plt.cm.Blues
    title = "Confusion Matrix"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



####################
##### LightGBM #####
####################
    
params = {'n_estimators': 100,
          'num_leaves': 48,
          'max_depth': 6,
          'subsample': 0.75,
          'learning_rate': 0.1,
          'min_child_samples': 8,
          'seed': 32,
          'nthread': -1}

lclf = lgb.LGBMClassifier(**params)
lclf.fit(x_train, y_train)
lclf.score(x_test, y_test)
    
# This results in test accuracy of 86.8%.    
# Given this library also has many parameters, similar to XGBoost, we need to use a similar strategy 
# of tuning in stages. First we will fix learning rate to a reasonable value of 0.1 and number of 
# estimators to be 200, then tune only the major tree building parameters
    
independent_params = {
    'seed': 32,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'nthread': -1}

params = {'max_depth': (4, 6, 8),
          'subsample': (0.75, 0.8, 0.9, 1.0),
          'colsample_bytree': (0.75, 0.8, 0.9, 1.0),
          'num_leaves': (12, 16, 36, 48, 54, 60, 80, 100)}

clf2 = EvolutionaryAlgorithmSearchCV(estimator=lgb.LGBMClassifier(**independent_params),
                                   params=params,
                                   scoring="accuracy",
                                   cv=5,
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=5,
                                   generations_number=100,
                                   n_jobs=-1)
clf2.fit(x_train, y_train)

# This gives the following set of optimal parameters:
# Best individual is: {'max_depth': 6, 'subsample': 1.0, 'colsample_bytree': 0.75, 'num_leaves': 54}
# with fitness: 0.870888486225853

# Now, we can use grid search to fine tune the search of number of leaves parameter.

independent_params = {
    'seed': 32,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'nthread': 1,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 0.75}

params = {'num_leaves': (48, 50, 52, 54, 56, 58, 60)} 
clf2 = GridSearchCV(lgb.LGBMClassifier(**independent_params), params, cv=5, n_jobs=-1, verbose=1)
clf2.fit(x_train, y_train)
print(clf2.best_params_)
    
# Similar to XGBoost, LightGBM also provides a cv() method that can be used to find optimal value of 
# number of estimators using early stopping strategy. 
#
# We find that the optimal value of n_estimators to be 327.
# Now, we can use the similar strategy to find and fine-tune the best regularization parameters.

independent_params = {
    'seed': 32,
    'learning_rate': 0.1,
    'n_estimators': 327,
    'nthread': 1,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 0.75,
    'num_leaves': 54}

params = {'reg_alpha' : [0,0.1,0.5,1],'reg_lambda' : [1,2,3,4,6],}
clf2 = GridSearchCV(lgb.LGBMClassifier(**independent_params), params, cv=5, n_jobs=-1, verbose=1)
clf2.fit(x_train, y_train)
print(clf2.best_params_)

# Finally, we can decrease the learning rate to 0.01 and find the optimal value of n_estimators.

independent_params = {
    'seed': 32,
    'learning_rate': 0.01,
    'nthread': 1,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 0.75,
    'num_leaves': 54,
    'reg_alpha': 0.62,
    'reg_lambda': 1.9}

params = {'n_estimators': (3270,3280,3300,3320,3340,3360,3380,3400)}

clf2 = GridSearchCV(lgb.LGBMClassifier(**ind_params), params, cv=5, n_jobs=8, verbose=1)
clf2.fit(x_train, y_train)
print(clf2.best_params_)
    
# We find the optimal n_estimators to be equal to 3327 for a learning rate of 0.01. We can now built a 
# final LightGBM model using these parameters and evaluate the test data.
    
independent_params = {
    'seed': 32,
    'learning_rate': 0.01,
    'n_estimators': 3327,
    'nthread': 8,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 0.75,
    'num_leaves': 54,
    'reg_alpha': 0.62,
    'reg_lambda': 1.9}

lclf = lgb.LGBMClassifier(**independent_params)
lclf.fit(x_train, y_train)
lclf.score(x_test, y_test)
    
y_pred = lclf.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

importances = lclf.feature_importances_
indices = np.argsort(importances)
cols = [cols[x] for x in indices]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), cols)
plt.xlabel('Relative Importance') 
