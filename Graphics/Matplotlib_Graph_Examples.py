
# coding: utf-8

# We'll use pandas to input the data and manage it so we'll need to import that package.

# In[41]:

import pandas as pd


# The statement below causes the matplotlib graphs to appear within the Jupter notebook rather than in a separate window that pops up.  This can be very convenient because, among other reasons, the image will be retained in the notebook and computation will not be paused when many graphs are constructed within a loop.  Statements that start with '%' are called magic functions.  I think it is also fair to say that these are system-level commands that you might type in a DOS Command Window.  The '%' indicates a system command so that it is executed properly.  This web page talks about magis functions: https://stackoverflow.com/questions/20961287/what-is-pylab

# In[42]:

get_ipython().magic(u'matplotlib inline')


# These are the data we will be working with using the pandas DataFrame data type

# df is often used as a variable name to denote a entity of the pandas DataFrame type

# The pandas .read_csv() method is very useful.  Note that the data files must have column headings in the first row.  You'll, possibly, need to adjust the path to the location of your data file if it is not in the same folder as your default Jupyter folder.
# 
# The first data set below (stored in the df_oz DataFrame) is from William S. Cleveland and it is included in a typical R installation, whcih is my sources.  These data describe temperature, wind speed, ozone, and radiation levels over time.  The second data set, which is from the US Census web site, gives time series data for public (governmental) construction spending and private construction spending.  The last data set, which is stored in the df_test DataFrame, gives high school standard of learning (SOL) scores for six course sections of a social studies course.  

# In[70]:

df_oz = pd.read_csv('C:\Users\Jing\Desktop\Competing Through Business Analytics\Storytelling & Visualization\data\ozone.csv')
df_oz.head()


# In[71]:

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.scatter(df_oz['wind'], df_oz['ozone'],alpha = 0.5)   #alpha controls the transparency of the dots
fig.suptitle('ozone vs. wind speed')
ax.yaxis.set_label_text('ozone')
ax.xaxis.set_label_text('wind speed')
fig.set_size_inches(7,5)
plt.show()


# In[72]:

fig,ax = plt.subplots()
ax.scatter(df_oz['radiation'], df_oz['ozone'], alpha = 0.5)
fig.suptitle('ozone vs. radiation')
ax.yaxis.set_label_text('ozone(ppb)')
ax.xaxis.set_label_text('radiation(langleys)')
ax.set_ylim(0,200)
ax.set_xlim(0,400)
fig.set_size_inches(7,5)
plt.show()


# In[83]:

""" Format of the scatterplot method is as follows: plt.scatter(x-series, y-series) """
plt.figure(figsize=(8,5))
plt.scatter(df_oz['wind'], df_oz['ozone'], alpha=0.5)
plt.plot(np.unique(df_oz['wind']), np.poly1d(np.polyfit(df_oz['wind'], df_oz['ozone'],3))(np.unique(df_oz['wind'])))
# alpha is a parameter that controls the transparency of the dots: 1 = solid, <1 = various transparency levels, 0 = no mark
plt.title('Ozone vs. Wind Speed')
plt.xlabel('Wind Speed')
plt.ylabel('Ozone')
plt.show()


# In[79]:

np.polyfit(df_oz['wind'], df_oz['ozone'],3)


# In[46]:

df_test = pd.read_csv('C:\\Users\\Jing\\Desktop\\Competing Through Business Analytics\\Storytelling & Visualization\\data\\test.csv')
df_test.head()


# In[47]:

import matplotlib.pyplot as plt


# Let's look at some distributional data from a database that contains student scores for two tests, one taken at the beginning of the year and one at the end of the year.  Each student's scores are recorded along with their isntructor, school number, and course section number.  Here are the column names:

# In[48]:

df_test.columns.values


# Before we proceed, let's clean things up a bit by using the StudentIdentifier data as the index.  It is unnecessary to have both the (automatic) integer index and the StudentIdentifier.

# In[49]:

df_test1 = df_test.set_index('StudentIdentifier')


# In[50]:

df_test1.head()


# pandas provides an easy way, using the .unique() method, to find the distinct entries in each column, for example to find what instructors are represented in the database

# In[51]:

df_test1['InstructorName'].unique()


# Let's filter the rows, choosing only those rows where Smith is the instructor

# In[52]:

df_test1.loc[df_test1['InstructorName'] == 'Smith']


# Let's plot a frequency histogram of this data for the year end test scores.

# In[53]:

df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore']


# Here's a frequency histogram for Instructor Smith's students at the end of the school year.

# In[54]:

fig, ax = plt.subplots()
ax.hist(x = df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore'], bins = 20, facecolor='g', alpha=0.75)

fig.suptitle('End of Year test Score Frequency Histogram')
fig.set_size_inches(7,5)

ax.xaxis.set_label_text('End Year Test Score')
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 

ax.yaxis.set_label_text('Frequency of Scores')
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off

ax.set_xlim(64, 85)
ax.set_ylim(0, 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(True)


# Here's one way to show a comparison of Smith's and Green's student scores at the end of the year.  Do you like the graph?  Is it easy to read?  Do you ahve any suggestions for improvement?

# In[55]:

# histogram is a bad way to show a side by side graph comparison


# In[56]:

fig, ax = plt.subplots()

ax.hist(x = df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore'], bins = 20, facecolor='r')
ax.hist(x = df_test1.loc[df_test1['InstructorName'] == 'Green']['EndYearTestScore'], bins = 20, facecolor='g')

fig.suptitle('End of Year test Score Frequency Histogram')
fig.set_size_inches(7,5)

ax.xaxis.set_label_text('End Year Test Score')
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 

ax.yaxis.set_label_text('Frequency of Scores')
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off

ax.set_xlim(64, 85)
ax.set_ylim(0, 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(True)


# In[57]:

fig, ax = plt.subplots()

ax.hist(x = df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore'], bins = 20, facecolor='r', alpha=0.5)
ax.hist(x = df_test1.loc[df_test1['InstructorName'] == 'Green']['EndYearTestScore'], bins = 20, facecolor='g', alpha=0.5)

fig.suptitle('End of Year test Score Frequency Histogram')
fig.set_size_inches(7,5)

ax.xaxis.set_label_text('End Year Test Score')
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 

ax.yaxis.set_label_text('Frequency of Scores')
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off

ax.set_xlim(64, 85)
ax.set_ylim(0, 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(True)


# In[58]:

fig, ax = plt.subplots()

n, bins, patches = ax.hist((df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore'],          df_test1.loc[df_test1['InstructorName'] == 'Green']['EndYearTestScore']), bins = 20, stacked = False)
for i in range(len(patches)):
    if i%2 == 0:
        plt.setp(patches[i],'facecolor','r')
    else:
        plt.setp(patches[i],'facecolor','g')

fig.suptitle('End of Year test score Frequency Histogram')
fig.set_size_inches(7,5)

ax.xaxis.set_label_text('End Year Test Score')
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 

ax.yaxis.set_label_text('Frequency of Scores')
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off

ax.set_xlim(64, 85)
ax.set_ylim(0, 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(True)
ax.legend(['Smith','Green'], loc=2)


# In[39]:

import numpy as np


# In[28]:

df_test2 = pd.concat([df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore'],df_test1.loc[df_test1['InstructorName'] == 'Green']['EndYearTestScore']], axis = 1)
df_test2.columns = ['Smith','Green']


# In[30]:

df_test2


# Here's a boxplot of Smith's end of year student test scores.

# In[31]:

# plt.boxplot will not take pandas dataframe data so data must be converted either to numpy array or a list
# the .as_matrix() method converts to a numpy array
# list() will convert a pandas series to a Python list
fig, ax = plt.subplots()

ax.boxplot(df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore'].as_matrix())

ax.yaxis.axes.set_ylim(60,85)
ax.set_xticklabels(['Smith'])

# Reducing clutter
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# Code for comparing Smith and Green

# In[32]:

# plt.boxplot will not take pandas dataframe data so data must be converted either to numpy array or a list
# the .as_matrix() method converts to a numpy array
# list() will convert a pandas series to a Python list

# Get data
data = []       # create an empty Python list; sublists will be appended for each boxplot
data.append(list(df_test1.loc[df_test1['InstructorName'] == 'Smith']['EndYearTestScore']))
data.append(list(df_test1.loc[df_test1['InstructorName'] == 'Green']['EndYearTestScore']))
data_min = min([min(sublist) for sublist in data])  # this and the following 3 lines automatically size the graph to the data 
data_max = max([max(sublist) for sublist in data])  # while providing buffer space

fig, ax = plt.subplots()

fig.set_figheight(5)
fig.set_figwidth(7)
fig.add_axes
ax.boxplot(data)

ax.yaxis.axes.set_ylim(data_min - 2, data_max + 2)
ax.set_xticklabels(['Smith','Green'])
ax.set_ylabel('Test Score')
ax.set_xlabel('Instructor')

# Reducing clutter
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# Code for comparing Smith, Jones, and Green

# In[33]:

# plt.boxplot will not take pandas dataframe data so data must be converted either to numpy array or a list
# the .as_matrix() method converts to a numpy array
# list() will convert a pandas series to a Python list

# Get data
instructors = ['Smith','Green','Jones']
data = []       # create an empty Python list; sublists will be appended for each boxplot
for instructor in instructors:
    data.append(list(df_test1.loc[df_test1['InstructorName'] == instructor]['EndYearTestScore']))
data_min = min([min(sublist) for sublist in data])  # this and the following 3 lines automatically size the graph to the data 
data_max = max([max(sublist) for sublist in data])  # while providing buffer space

fig, ax = plt.subplots()

fig.set_figheight(5)
fig.set_figwidth(7)
fig.add_axes

ax.boxplot(data)

ax.yaxis.axes.set_ylim(data_min - 2, data_max + 2)
ax.set_xticklabels(instructors)
ax.set_ylabel('Test Score')
ax.set_xlabel('Instructor')
ax.set_xticks(range(1,len(instructors)+1),instructors)

# Reducing clutter
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# In[34]:

data = []       # create an empty Python list; sublists will be appended for each boxplot
instructors = ['Smith','Green','Jones']
for instructor in instructors:
    data.append(list(df_test1.loc[df_test1['InstructorName'] == instructor]['EndYearTestScore']))
plt.boxplot(data)
plt.xticks(range(1,len(instructors)+1),instructors)
data_min = min([min(sublist) for sublist in data])  # this and the following 3 lines automatically size the graph to the data 
data_max = max([max(sublist) for sublist in data])  # while providing buffer space
plt.ylim([data_min - 2, data_max + 2])   
plt.show()


# In[59]:

# Pareto charts have a standard format
import matplotlib.pyplot as plt

# Data
bdata = [1.28,1.05,0.6093,0.22195,0.16063,0.1357,0.10226,0.08499,0.06148,0.05022,0.04485,0.02981]
blabels = ['Unemp','Health','Mil.','Interest', 'Veterans','Agri.','Edu','Trans','Housing','Intl','EnergyEnv','Science']
xs = range(len(bdata))
bdata_cum = []
for i in range(len(bdata)):
    bdata_cum.append(sum(bdata[0:i+1])/sum(bdata))

fig, ax = plt.subplots()

# Set bar chart parameters
ax.bar(xs,bdata, align='center')
ax.set_ylim(0,sum(bdata))
ax.set_xticks(xs)
ax.set_xticklabels(blabels, rotation = 45)
ax.grid(False)

ax.tick_params(axis = 'y', which = 'both', direction = 'in', width = 2, color = 'black')

# Set line chart paramters and assign the second y axis
ax1 = ax.twinx()
ax1.plot(xs,bdata_cum,color='k')
ax1.set_ylim(0,1)
ax1.set_yticklabels(['{:1.1f}%'.format(x*100) for x in ax1.get_yticks()])
ax1.grid(False)

fig.set_figwidth(9)
fig.set_figheight(5)


# In[60]:

# introduce 'other' category for long tail of small measurements
import matplotlib.pyplot as plt

# Data
blabels1 = ['SS','Health','Mil.','Interest', 'Vet.','Agri.','Other']
bindex = 6
bother = sum(bdata[bindex:])
bdata1 = bdata[:bindex] + [bother]
xs = range(len(bdata1))
bdata_cum = []
for i in range(len(bdata1)):
    bdata_cum.append(sum(bdata1[0:i+1])/sum(bdata1))

fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(5)

# Bar chart settings
ax.set_xticks(xs)
ax.set_xticklabels(blabels1)
ax.bar(xs,bdata1, align='center')
ax.set_ylim(0,sum(bdata1))

# Line chart settings
ax1 = ax.twinx()
ax1.plot(xs,bdata_cum,color='k')
ax1.set_ylim(0,1)
ax1.set_yticklabels(['{:1.1f}%'.format(x*100) for x in ax1.get_yticks()])


# In[61]:

import matplotlib.pyplot as plt

bdata_cum = []
for i in range(len(bdata1)):
    bdata_cum.append(sum(bdata1[0:i+1])/sum(bdata1))
fig, ax = plt.subplots()
ax.set_xticklabels(['']+blabels1)
xs = range(len(bdata1))
ax.bar(xs,bdata1, align='center')
ax.set_ylim(0,sum(bdata1))
ax1 = ax.twinx()
ax1.plot(xs,bdata_cum,color='k')
ax1.set_ylim(0,1)
ax1.set_yticklabels(['{:1.1f}%'.format(x*100) for x in ax1.get_yticks()])
plt.show()


# In[66]:

# using Pareto package
pdata = [21, 2, 10, 4, 16]
plabels = ['tom', 'betty', 'alyson', 'john', 'bob']
from paretochart import pareto
pareto(pdata, plabels, limit=0.95, line_args=('g',))


# We will create a violin plot of this data using the seaborn package.
# The seaborn package interoperates with pandas data better than does matplotlib.

# In[67]:

import seaborn as sns


# The line below will cause the graphs to appear within the jupyter window

# In[3]:

get_ipython().magic(u'matplotlib inline')


# In[37]:

#f, ax = plt.subplots(figsize=(8, 8))
sns.set_style("whitegrid")
sns.despine(left=True)
fig = sns.violinplot(x="InstructorName",y="EndYearTestScore",data=df_test1, inner="box", palette="Set3", cut=2, linewidth=3)
fig.set_xlabel("Instructor",size = 16,alpha=0.7)
fig.set_ylabel("Test Score",size = 16,alpha=0.7)
# note that matplotlib is operating in the background here and so we can use it to save teh figure to file
plt.savefig('instructors.png')


# seaborn is written 'on top of' matplotlib and automates many forms of multi-graph figures.  Here's is one example where each one-to-one relationships in the four fields in the ozone data are shown via a grid of scatter plots.  seaborn in this way takes care of a lot of detailed work that you would need to do if you used matplotlib to construct this graph.

# In[184]:

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df_oz)
g.savefig('ozone.jpg')


# In[79]:

import seaborn as sns
sns.set(style="whitegrid")

xs = [0,82.0442626658045,164.140532825633,249.101916493353,332.182520454561,416.254670963008,495.330844802651,575.152803567278,653.205144040749,738.002602107511,818.027625347191,898.118246791891,980.079475188722,1061.14169245002,1139.27372549306,1222.46124841828,1300.35875796241,1375.56916894815,1450.58381247379,1528.61926864683,1605.58397275636,1690.73526419816,1765.8607155748,1844.76374052086,1927.68455634244,2007.86892722346,2086.83606915306,2166.71913598734,2251.80135023691,2335.70816516882,2420.79263834332,2499.86354911873,2577.05890165442,2654.19208140119,2730.09283946298,2809.07577275313,2894.08624312766,2969.05670691318,3049.93958564477,3129.75861740531,3206.68277248806,3283.60346903165,3364.58552562287,3449.40362446521,3525.4406380069,3608.13685182861,3690.05929028829,3773.14656467608,3850.22786804579,3930.30737684189,4015.46434790825,4093.50879464788,4172.64841837878,4252.66450124259,4335.57956650011,4420.52164467734,4495.52991250898,4575.39742270308,4650.49642074232,4730.52574677055,4805.68851119975,4880.60019974235,4960.70865715944,5043.67168991259,5118.73724644963,5203.54753389807,5283.38430140739,5359.51383129058,5438.3504224767,5519.19106748314,5597.38879699402]
ys = [69,70.5331166725557,71.5580608629997,71.7925736363559,71.3468902075964,70.8280365103217,70.6414982437025,70.4451053764375,70.3795952412711,69.7645648576213,69.5426154537094,69.3004779067882,68.9026747371149,68.599154906399,68.519069857217,68.3772433121902,69.1486631584102,68.7548264957532,68.3609585242701,68.5575021969504,68.5786288310157,70.1801398605569,69.7914710297875,70.1491959166127,71.3308639026079,71.9361568322099,71.7841530239482,71.5632100706621,70.9532872882161,70.4120650812618,69.7956890967353,69.6193830861858,69.6154149236862,69.6130427165012,69.7102463368457,69.5264209638887,68.9050613087613,69.1051712486311,68.8193037564598,68.6066819000086,68.6092749335593,68.6204785117934,68.3360360051156,68.5812312774717,68.3713770000006,69.5440095449469,70.5260768620176,72.0285425812187,72.0349254824144,71.8158332177273,71.1892654017154,71.1169346409977,70.8948191898231,70.6895661675202,70.2298913978404,69.6255849641529,69.7340391269025,69.5070018796742,69.6617353391113,69.4627544694931,69.630774569752,69.8026157834872,69.5269217792645,69.0549816241028,69.1984712324886,68.3562401007611,68.1422906748129,68.2062726166005,68.0498033914326,68.6018257745021,68.7606159741864]


#import seaborn as sns
sns.set(style="darkgrid", color_codes=True)

tips = sns.load_dataset("tips")
g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",
                  xlim=(0, 60), ylim=(0, 12), color="r", size=7)

