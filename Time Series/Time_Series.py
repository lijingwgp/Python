
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

get_ipython().magic(u'matplotlib inline')


# In[3]:

df1 = pd.read_csv('C:\Users\Jing\OneDrive\Class\Competing Through Business Analytics\Data Analysis with pandas\TSDataP1.csv')
df1.head()


# Explain pandas. pandas is one of the most frequently used Python packages and it provides extremely functional data types and methods to handle data.  Some of its key features are as follows:
#     * The two main data types are pandas Series and pandas DataFrame.  The former have one 
#       column of data and the latter have multiple columns of data.  Both data types have an 
#       index column, which can be used to refer to the rows of the dataData columns.
#     * Columns in the DataFrame data type objects have names which make the data more understandable.
#     * The data columns in a Series object also has a name
#     * Data cleansing operations are easy with methods such as .fillna() and .dropna(), which handle null data elements
#     * Database-type operations can be performed, such as joins.
#     * DataFrames permit the concatenation of additional columns
#     * The DataFrame.apply() method is very useful to define new DataFrame columns or alter data in 
#       existing columns.

# In[4]:

df1.shape


# In[5]:

df1.describe()


# In[6]:

df1.info()


# In[7]:

df1.columns.values


# In[9]:

df1.index


# In[10]:

import matplotlib.pyplot as plt
plt.plot(df1.index,df1.Product1Demand)


# At this point you would notice a (postiive) trend  and seasonality.  You could, and should ask yourself what is causing those patterns.  You may know the causes from experience if you are familiar with the context or, alternately, you may need to do research to determine the causes.  If you were a consultant, you might well be in the latter position.

# In[11]:

df2 = pd.read_csv('C:\Users\Jing\OneDrive\Class\Competing Through Business Analytics\Data Analysis with pandas\TSDataP2.csv')
df2.head()


# In[12]:

import matplotlib.pyplot as plt
plt.plot(df2.index,df2.Product2Demand)


# In[13]:

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df1.index,df1.Product1Demand)
ax.plot(df2.index,df2.Product2Demand)
fig.figsize = (12,8)


# In[14]:

import numpy as np

print np.corrcoef(df1.Product1Demand, df2.Product2Demand)


# In[17]:

from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(df1.Quarter,df1['Product1Demand'])
print 'intercept =', intercept, '    slope =', slope


# In[19]:

def create_regress_col(row, intercept, slope):
    return float(intercept) + float(row[0]) * slope
    
df1['regress'] = df1.apply(create_regress_col,args = (intercept,slope),axis = "columns")
df1


# In the computation below, a column is created in the DataFrame whose name is 'R1'.  The 'R' stands for remainder of the original sales trajectory after the regression 'pattern' has been extracted from the original sales pattern.

# In[20]:

df1['R1'] = df1['Product1Demand'] - df1['regress']
df1


# Here is a plot of the R1 data column.

# In[21]:

import matplotlib.pyplot as plt
plt.plot(df1.index,df1.R1)


# When the autocorrelation value for a particular lag is large (close to 1) and positive, it indicates a cyclic pattern with the periodicty of that lag.

# In[22]:

for i in range(len(df1.index)/2):
    print 'autocorrelation, lag =',i,':',df1.R1.autocorr(lag = i)


# This code plots each sequential series of 4 points, where 4 corresponds with the periodicty of the data.  Note how the patterns have similar shapes, which is why the autocorrelation with this lag was nearly 1.

# In[23]:

import matplotlib.pyplot as plt

i = 0
cycle = 4
df_unstack = pd.DataFrame()
while 4*i < len(df1['Product1Demand']):
    new_df = pd.DataFrame(data = {i:list(df1['Product1Demand'][i*4:min(len(df1['Product1Demand']),i*4+4)])},index = range(1,cycle+1))
    #print new_df
    df_unstack = pd.concat([df_unstack,new_df],axis=1)
    #print "\n", df1['Product1Demand'][i*4+1:min(len(df1['Product1Demand']),i*4+4+1)]
    #print df_unstack
    i = i+1   
df_unstack

for col in df_unstack.columns:
    plt.plot(df_unstack.index, list(df_unstack[col]), label = col)
    
plt.legend(loc=8)
slope, intercept, r_value, p_value, std_err = stats.linregress(df1.index,df1['Product1Demand'])
print intercept, slope


# The code below uses a simple method of estimating the average sales in each period by summing the sales in the time series corresponding to each 'season' and dividing by the number of points to arrive at an average sales figure for each season.  A new column, names 'S', is added to the DataFrame.

# In[24]:

def season(row,cycle,S):
    #print row[0]
    return S[int(row[0])%cycle - 1]

S = []
for i in range(cycle):
    this_list = df1.iloc[i::4, :].R1
    S.append(sum(this_list)/float(len(this_list)))
df1['S'] = df1.apply(season,args = (cycle,S), axis = "columns")
S


# In[25]:

df1


# Similarly to when we created the dataFrame column names 'R1', we create a DataFrame column in the code below named 'R2' for the second remainder, this time after we extract the seasonal/cyclic component from the original sales data.  'R2' represents the part of the data that we ahve not explained with a constant, a trend, and seasonality.  In other words, this is the 'error' if we were using the first three components as a forecast.

# In[26]:

df1['R2'] = df1['R1'] - df1['S']
df1


# Just to see how much of the original data is included in our regression and seasonality components, let's compute the percentage of the final remainder R2 as a percentage of the iringal sales quantity.

# In[59]:

df1.info()


# In[ ]:




# In[61]:

df1['ErrPerc'] = abs(df1.R2/df1.Product1Demand.astype(float))
df1


# In[27]:

import matplotlib.pyplot as plt
plt.plot(df1.index,df1.R2)


# In[30]:

for i in range(5):
    print 'Correlation of lag ', i, ': ',df1.R2.autocorr(lag = i)


# The series of cells below performs the same decomposition on the Product 2 sales data.

# In[31]:

from scipy import stats

slopeR2, interceptR2, r_valueR2, p_valueR2, std_errR2 = stats.linregress(df1.Quarter,df1['R2'])
print 'intercept =', interceptR2, '    slope =', slopeR2, '     r_value = ',r_valueR2, '     p_value =',p_valueR2


# In[32]:

from scipy import stats

slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df2.Quarter,df2['Product2Demand'])
print 'intercept =', intercept2, '    slope =', slope2


# In[35]:

def create_regress_col(row, intercept, slope):
    return float(intercept) + float(row[0]) * slope
    
df2['regress'] = df2.apply(create_regress_col,args = (intercept,slope),axis = "columns")
df2['R1'] = df2['Product2Demand'] - df2['regress']
df2


# In[34]:

def season(row,cycle,S):
    return S[int(row[0])%cycle - 1]

S2 = []
for i in range(cycle):
    this_list = df2.iloc[i::4, :].R1
    S2.append(sum(this_list)/float(len(this_list)))
df2['S'] = df2.apply(season,args = (cycle,S2), axis = "columns")
S2


# In[36]:

df2['R2'] = df2['R1'] - df2['S']
df2


# In[62]:

df2['ErrPerc'] = abs(df2.R2/df2.Product2Demand.astype(float))
df2


# The cell below computes the correlation of the Product 1 seasoanlity component with the Product 2 seasonality component.  It includes an example of how you can format output.

# In[56]:

import numpy as np

corr_mat =  np.corrcoef(df1.S, df2.S)
print type(corr_mat),'\n',corr_mat
indices = range(len(corr_mat))

for i in indices:
    if i == 0:
        print "     ",
        for j in indices:
            print  "%9s" % j,
        print
    print "    %s" % (i),
    for j in indices:
        print "    %7.5f" % corr_mat[i][j],
    print


# Here's the same type of correlation analysis for the regression portion of the sales data decomposition.

# In[63]:

import numpy as np

corr_mat =  np.corrcoef(df1.regress, df2.regress)
print type(corr_mat),'\n',corr_mat
indices = range(len(corr_mat))

for i in indices:
    if i == 0:
        print "     ",
        for j in indices:
            print  "%9s" % j,
        print
    print "    %s" % (i),
    for j in indices:
        print "    %7.5f" % corr_mat[i][j],
    print

Conclusions

    * The demands for both products are increasing with time and exhibit cyclicality, 
      which can be called seasonality in this case because its period is four quarters.
    * Both products demand can be decomposed into four components: a constant, a trend, 
      seasonality, and the remaining noise/error.
    * The peak sales of Product 1 are in Quarters 2 and 3, Spring and Summer, implying a 
      product such as related to warm season recreation or home care.
    * The peak sales of Product 2 are in Quarters 3 and 4.  This product may be related
      Fall and Winter activities.
    * By definition the two products demand trends are positively correlated with a 
      coefficient of 1.0
    * The seasonality components of the the two products' demands are negatively
      correlated with a coefficient of -0.94.
# In[ ]:



