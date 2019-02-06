# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 12:44:05 2019

@author: jing.o.li
"""

import math 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x = load_breast_cancer().data
y = load_breast_cancer().target
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=31)

lasso = Lasso()
lasso.fit(x_train,y_train)
lasso_test_score = lasso.score(x_test, y_test)
coefficients = np.sum(lasso.coef_ != 0)

lasso001 = Lasso(alpha=.01, max_iter=10e5)
lasso001.fit(x_train, y_train)
lasso001_test_score = lasso001.score(x_test, y_test)
coefficients1 = np.sum(lasso001.coef_ != 0) 

lasso00001 = Lasso(alpha=.0001, max_iter=10e5)
lasso00001.fit(x_train,y_train)
lasso00001_test_score = lasso00001.score(x_test, y_test)
coefficients2 = np.sum(lasso00001.coef_ != 0)

lr = LinearRegression()
lr.fit(x_train,y_train)
lr_test_score=lr.score(x_test,y_test)

plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)

plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.00001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.show()

# Let's understand the plot and the code in a short summary
# 
# the default value of regularization parameter in Lasso is 1
# with lambda value of 1, only 4 features are used
# with only 4 features, both training and test score are low, the model is under-fitting
#
# reduce the lambda to 0.01, non-zero features increased to 10, training and test
# scores also increased.
#
# with lambda = 1, we see that most of the coefficients are zero or nearly zero
#
# further reduce lambda to 0.0001, non-zero features = 22. training and test score
# are similar to basic linear regression case. 
#
# with lambda = 0.0001, coefficients for Lasso regression and linear regression 
# show close resemblance.
