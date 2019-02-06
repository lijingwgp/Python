# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:58:41 2019

@author: jing.o.li
"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston = load_boston()
boston = pd.DataFrame(boston.data, columns=boston.feature_names)
boston.info()
target = load_boston().target
x_train, x_test, y_train, y_test = train_test_split(boston, target, test_size=0.3, random_state=3)

lr = LinearRegression()
lr.fit(x_train, y_train)

# higher the alpha value, more restriction on the coefficients
# low alpha value leads to more generalization, coefficients are barely
# restricted and in this case linear and ridge regression resembles
rr = Ridge(alpha=.01)
rr.fit(x_train, y_train)

# now, let's try a higher alpha value
rr100 = Ridge(alpha=100)
rr100.fit(x_train, y_train)

# let's compare results
linear_test_score = lr.score(x_test, y_test)
ridge_test_score = rr.score(x_test, y_test)
ridge100_test_score = rr100.score(x_test, y_test) 

plt.plot(rr.coef_, alpha=.7, linestyle='none', marker='*', markersize=5, color='red',
         label=r'Ridge; $\alpha = 0.01$', zorder=7)
plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') 
plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.show()

# In X axis we plot the coefficent index and for Boston data, there are 13 features.
# Fow low value of alpha (0.01), when the coefficients are less restricted, the coefficient
# magnitudes are almost same as of linear regression. 
#
# For higher value of alpha (100), we see that for coefficient indices 3,4,5 the magnitudes
# are considerably less compared to linear regression case. 
