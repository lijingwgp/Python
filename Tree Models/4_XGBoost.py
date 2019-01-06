# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:35:57 2019

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



###################
##### XGBoost #####
###################

params = {'n_estimators': 100,
          'max_depth': 6,
          'subsample': 0.75,
          'learning_rate': 0.1,
          'min_samples_split': 2,
          'min_samples_leaf': 8,
          'random_state': 32,
          'objective': 'binary:logistic',
          'n_jobs': -1}

xclf = xgb.XGBClassifier(**params)
xclf.fit(x_train, y_train)
xclf.score(x_test, y_test)
    
# With this reasonable guess of parameters based on previous models, we already see an accuracy of 86.75%.
# Let us try to find the optimal parameters for the XGBoost model. If we simply try to do a brute force 
# grid search, it can be computationally very expensive and unreasonable on a desktop. Here is a 
# sample parameters list that can give us an idea of what such a grid search could look like.   

independent_params = {
    'random_state': 32,
    'objective': 'binary:logistic',
    'n_jobs': -1,}

params = {'n_estimators': (100, 200, 400, 800, 1000),
          'max_depth': (4, 6, 8),
          'subsample': (0.75, 0.8, 0.9, 1.0),
          'learning_rate': (0.1, 0.01, 0.05),
          'colsample_bytree': (0.75, 0.8, 0.9, 1.0),
          'min_child_weight': range(1,6,2),
          'reg_alpha': [i/10.0 for i in range(0,5)],
          'gamma':[i/10.0 for i in range(0,5)],
          'reg_lambda': (1, 0.1, 10)}

# If we try to do a grid search of this with 5-fold cross validation, it will involve a whopping 0.81 
# million model training calls! And, even this will not be enough, as we will need additional model 
# training steps to fine-tune our parameter search for finer and/or different range of parameters. 
# Clearly, we need a different approach to solve this.
    
    

##########################
##### XGBoost Tuning #####
##########################

# I will take an approach of optimizing different set of parameters in batches. To begin, we will choose 
# a fixed learning rate of 0.1, and n_estimators=200. We will try to find only tree related parameters
# (i.e. max_depth, gamma, subsample and colsample_bytree) using grid search or genetic algorithm.

independent_params = {
    'random_state': 32,
    'objective': 'binary:logistic',
    'n_estimators': 200,
    'learning_rate': 0.1}

params = {'max_depth': (4, 6, 8),
          'subsample': (0.75, 0.8, 0.9, 1.0),
          'colsample_bytree': (0.75, 0.8, 0.9, 1.0),
          'min_child_weight': range(1,6,2),
          'gamma': [i/10 for i in range(0,5)]}

clf2 = EvolutionaryAlgorithmSearchCV(estimator=xgb.XGBClassifier(**independent_params),
                                   params=params,
                                   scoring="accuracy",
                                   cv=5,
                                   verbose=1,
                                   population_size=60,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=5,
                                   generations_number=100,
                                   n_jobs=-1)
clf2.fit(x_train, y_train)

# This gives us the following optimal values for different parameters:
# Best individual is: {'max_depth': 6, 'subsample': 1.0, 'colsample_bytree': 0.8, 'gamma': 0.2, 'min_child_weight': 1}
# with fitness: 0.8710727557507447
#
# XGBoost provides an optimized version of cross validation using cv() method which supports early 
# stopping to give us optimal value of n_estimators.

xgb1 = xgb.XGBClassifier(
 learning_rate=0.1,
 n_estimators=10000,
 max_depth=6,
 min_child_weight=1,
 gamma=0.2,
 subsample=1.0,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 n_jobs=-1,
 scale_pos_weight=1,
 random_state=32)

xgb_param = xgb1.get_xgb_params()
xgtrain = xgb.DMatrix(x_train, label=y_train)

cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5, 
                  metrics='auc', early_stopping_rounds=50)

# This gives us a value of n_estimators = 206. 
print("Number of Predicted n_estimators = ", cvresult.shape[0])

# We can now use these parameters as fixed values and optimize regularization parameters: reg_alpha and reg_lambda.

independent_params = {'learning_rate': 0.1,
 'n_estimators': 206,
 'gamma': 0.2,
 'subsample': 1.0,
 'colsample_bytree': 0.8,
 'objective': 'binary:logistic',
 'random_state': 32,
 'max_depth': 7,
 'min_child_weight': 1}

params = {'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05], 'reg_lambda':[0.01, 0.1, 1, 10, 100]}

clf2 = EvolutionaryAlgorithmSearchCV(estimator=xgb.XGBClassifier(**independent_params),
                                   params=params,
                                   scoring="accuracy",
                                   cv=5,
                                   verbose=1,
                                   population_size=30,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=100,
                                   n_jobs=8)
clf2.fit(x_train, y_train)

# The optimal set of parameters found by this search are:
# Best individual is: {'reg_alpha': 0.001, 'reg_lambda': 1}
# with fitness: 0.8714720063880101

# We can now decrease the learning rate by an order to magnitude to get a more stable model. 
# However, we will also need to find the optimal value of number of estimators again using the cv() method.

independent_params = {'learning_rate': 0.01,
 'n_estimators': 5000,
 'gamma': 0.2,
 'reg_alpha': 0.001,
 'reg_lambda': 1,
 'subsample': 1.0,
 'colsample_bytree': 0.8,
 'objective': 'binary:logistic',
 'random_state': 32,
 'max_depth': 7,
 'n_jobs': 8,
 'min_child_weight': 1}

xgb2 = xgb.XGBClassifier(**independent_params)
xgb_param = xgb2.get_xgb_params()
xgtrain = xgb.DMatrix(x_train, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb2.get_params()['n_estimators'], nfold=5, 
                  metrics='auc', early_stopping_rounds=50)

# We get an optimal value of n_estimators = 1559. Let us use now all of these optimized values to make 
# a final XGBoost model.
print("Number of Predicted n_estimators = ", cvresult.shape[0])



######################
##### Evaluation #####
######################

independent_params = {'learning_rate': 0.01,
 'n_estimators': 1559,
 'gamma': 0.2,
 'reg_alpha': 0.001,
 'reg_lambda': 1.0,
 'subsample': 1.0,
 'colsample_bytree': 0.8,
 'objective': 'binary:logistic',
 'random_state': 32,
 'max_depth': 7,
 'n_jobs': 8,
 'min_child_weight': 1}

xclf = xgb.XGBClassifier(**independent_params)
xclf.fit(x_train, y_train)
xclf.score(x_test, y_test)

y_pred = xclf.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

importances = xclf.feature_importances_
indices = np.argsort(importances)
cols = [cols[x] for x in indices]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), cols)
plt.xlabel('Relative Importance')
