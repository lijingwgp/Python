
# coding: utf-8

# In[11]:

import pandas as pd


# In[12]:

get_ipython().magic(u'matplotlib inline')


# In[13]:

df = pd.read_csv('C:\\Users\\Jing\\Desktop\\ConstructionTimeSeriesDataV2.csv')
df.head()
df.info()
df.describe()


# In[17]:

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(len(df['Total Construction'])), df['Total Construction'])
ax.xaxis.set_ticklabels(df['Month-Year'])


# In[20]:

print range(0, len(df['Total Construction']), 12)
print df['Month-Year'][::12]


# In[28]:

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(len(df['Total Construction'])), df['Total Construction'])
ax.set_xticks(range(0,len(df['Total Construction']), 12))
ax.xaxis.set_ticklabels(df['Month-Year'][::12])

fig.set_size_inches((10, 7))


# In[ ]:



