# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:30:15 2018

@author: jing.o.li
"""

############################
##### Data Preparation #####
############################

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from plotnine import *
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

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



#########################
##### Random Forest #####
#########################

rclf = RandomForestClassifier(n_estimators=500)
rclf.fit(x_train, y_train)
rclf.score(x_test, y_test)

# Even without any optimization, we find the model to be quite close to the optimized 
# tree classifier with a test score of 85.1%.
#
# In terms of the confusion matrix, we again find this to be quite comparable to the 
# optimized tree classifier with a prediction accuracy of 92.1% for the majority class 
# (<=50K Income) and a prediction accuracy of 62.6% for the minority class (>50K Income).

y_pred = rclf.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

# As discussed before, random forest models also provide us with a metric of feature 
# importances. We can see importance of different features in our current model as below:

importances = rclf.feature_importances_
indices = np.argsort(importances)
cols = [cols[x] for x in indices]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), cols)
plt.xlabel('Relative Importance')

# Now, let us try to optimize our random forest model

parameters = {
     'n_estimators':(100, 500, 1000),
     'max_depth':(None, 24, 16),
     'min_samples_split': (2, 4, 8),
     'min_samples_leaf': (16, 4, 12)
}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
clf.fit(x_train, y_train)
clf.best_score_, clf.best_params_

rclf2 = RandomForestClassifier(n_estimators=1000,max_depth=24,min_samples_leaf=4,min_samples_split=8)
rclf2.fit(x_train, y_train)
y_pred = rclf2.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

