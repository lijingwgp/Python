# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:22:39 2017

@author: Jing
"""

import numpy as np
import pandas as pd

df = pd.read_csv("PastHires.csv")         ## read data
df.head()                                 ## preview of data
df.head(10)
df.tail(4)
df.shape                                  ## data dimension
df.size                                   ## number of unique data points
len(df)                                   ## number of columns
df.columns                                ## column names
df['Hired']                               ## extract column
df['Hired'][:5]
df['Hired'][5]
df[['Hired','Years Experience']]          ## extract multiple column
df[['Hired','Years Experience']][:5]
df.sort_values(['Years Experience'])      ## sort dataframe
df['Level of Education'].value_counts()   ## break down the number of unique values in a column
