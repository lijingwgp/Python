# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:41:07 2018

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



###############
##### EDA #####
###############

# let's first see how relationships column and maritalstatus column are interrelated
(ggplot(df_train, aes(x="Relationship", fill="MaritalStatus")) + 
 geom_bar(position="fill") + 
 theme(axis_text_x = element_text(angle=60, hjust=1)))

# let's look at effect of Education on Income for different Age groups
(ggplot(df_train, aes(x = "Education", fill = "Income"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = 60, hjust = 1))
 + facet_wrap('~AgeGroup')
)

# next, let's look at the effect of Education and Race for males and females separately
(ggplot(df_train, aes(x = "Education", fill = "Income"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = -90, hjust = 1))
 + facet_wrap('~Sex')
)
(ggplot(df_train, aes(x = "Race", fill = "Income"))
 + geom_bar(position="fill")
 + theme(axis_text_x = element_text(angle = -90, hjust = 1))
 + facet_wrap('~Sex')
)

# until now, we have only looked at the inter-dependence of non-numeric features.
# let's now look at the effect of CapitalGain and CapitalLoss on income

(ggplot(df_train, aes(x="Income", y="CapitalGain"))
 + geom_jitter(position=position_jitter(0.1))
)
(ggplot(df_train, aes(x="Income", y="CapitalLoss"))
 + geom_jitter(position=position_jitter(0.1))
)



###########################
##### Tree Classifier #####
###########################

# now that we understand some relationship in our data, let us build a simple tree 
# classifier model. however, in order to use this module, we need to convert all of 
# our non-numeric data to numeric ones.

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

colnames = list(df_train)
colnames.remove("Income")
colnames = colnames[:-3] + ['Income'] + colnames[-3:]

df_train1 = mapper.fit_transform(df_train.copy())
df_train1.columns = colnames
df_test1 = mapper.fit_transform(df_test.copy())
df_test.columns = colnames

colnames.remove("Income")
x_train, y_train = df_train1[colnames].values, df_train1["Income"].values
x_test, y_test = df_test1[colnames].values, df_test1["Income"].values

treeClassifier = DecisionTreeClassifier()
treeClassifier.fit(x_train, y_train)
treeClassifier.score(x_test, y_test)

# The simplest possible tree classifier model with no optimization gave us an accuracy 
# of 83.5%. In the case of classification problems, confusion matrix is a good way 
# to judge the accuracy of models. 



##############################
##### Evaluation Results #####
##############################

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

# Now, we can take a look at the confusion matrix of out first model:
y_pred = treeClassifier.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

# We find that the majority class (<=50K Income) has an accuracy of 90.5%, 
# while the minority class (>50K Income) has an accuracy of only 60.8%.



##########################
##### Grid Search CV #####
##########################

# Let us look at ways of tuning this simple classifier. We can use GridSearchCV() 
# with 5-fold cross-validation to tune various important parameters of tree classifiers.

parameters = {
     'max_features':(None, 9, 6),
     'max_depth':(None, 24, 16),
     'min_samples_split': (2, 4, 8),
     'min_samples_leaf': (16, 4, 12)
}
clf = GridSearchCV(treeClassifier, parameters, cv=3, n_jobs=-1)
clf.fit(x_train, y_train)
clf.best_score_, clf.score(x_test, y_test), clf.best_params_

# With the optimization, we find the accuracy to increase to 85.9%. In the above, 
# we can also look at the parameters of the best model. Now, let us have a look at the 
# confusion matrix of the optimized model.

y_pred = clf.predict(x_test)
cfm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(10,6))
plot_confusion_matrix(cfm, classes=["<=50K", ">50K"], normalize=True)

