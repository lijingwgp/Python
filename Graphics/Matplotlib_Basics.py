
# coding: utf-8

# The statement below causes the matplotlib graphs to appear within the Jupter notebook rather than in a separate window that pops up.  This can be very convenient because, among other reasons, the image will be retained in the notebook and computation will not be paused when many graphs are constructed within a loop.  Statements that start with '%' are called magic functions.  I think it is also fair to say that these are system-level commands that you might type in a DOS Command Window.  The '%' indicates a system command so that it is executed properly.  This web page talks about magis functions: https://stackoverflow.com/questions/20961287/what-is-pylab

# In[1]:

get_ipython().magic(u'matplotlib inline')


# Paste the text from the file matplotlib_eg_line.py below and execute it (Ctrl+Enter).

# In[2]:

import matplotlib.pyplot as plt

xs = [0,82.0442626658045,164.140532825633,249.101916493353,332.182520454561,416.254670963008,495.330844802651,575.152803567278,653.205144040749,738.002602107511,818.027625347191,898.118246791891,980.079475188722,1061.14169245002,1139.27372549306,1222.46124841828,1300.35875796241,1375.56916894815,1450.58381247379,1528.61926864683,1605.58397275636,1690.73526419816,1765.8607155748,1844.76374052086,1927.68455634244,2007.86892722346,2086.83606915306,2166.71913598734,2251.80135023691,2335.70816516882,2420.79263834332,2499.86354911873,2577.05890165442,2654.19208140119,2730.09283946298,2809.07577275313,2894.08624312766,2969.05670691318,3049.93958564477,3129.75861740531,3206.68277248806,3283.60346903165,3364.58552562287,3449.40362446521,3525.4406380069,3608.13685182861,3690.05929028829,3773.14656467608,3850.22786804579,3930.30737684189,4015.46434790825,4093.50879464788,4172.64841837878,4252.66450124259,4335.57956650011,4420.52164467734,4495.52991250898,4575.39742270308,4650.49642074232,4730.52574677055,4805.68851119975,4880.60019974235,4960.70865715944,5043.67168991259,5118.73724644963,5203.54753389807,5283.38430140739,5359.51383129058,5438.3504224767,5519.19106748314,5597.38879699402]
ys = [69,70.5331166725557,71.5580608629997,71.7925736363559,71.3468902075964,70.8280365103217,70.6414982437025,70.4451053764375,70.3795952412711,69.7645648576213,69.5426154537094,69.3004779067882,68.9026747371149,68.599154906399,68.519069857217,68.3772433121902,69.1486631584102,68.7548264957532,68.3609585242701,68.5575021969504,68.5786288310157,70.1801398605569,69.7914710297875,70.1491959166127,71.3308639026079,71.9361568322099,71.7841530239482,71.5632100706621,70.9532872882161,70.4120650812618,69.7956890967353,69.6193830861858,69.6154149236862,69.6130427165012,69.7102463368457,69.5264209638887,68.9050613087613,69.1051712486311,68.8193037564598,68.6066819000086,68.6092749335593,68.6204785117934,68.3360360051156,68.5812312774717,68.3713770000006,69.5440095449469,70.5260768620176,72.0285425812187,72.0349254824144,71.8158332177273,71.1892654017154,71.1169346409977,70.8948191898231,70.6895661675202,70.2298913978404,69.6255849641529,69.7340391269025,69.5070018796742,69.6617353391113,69.4627544694931,69.630774569752,69.8026157834872,69.5269217792645,69.0549816241028,69.1984712324886,68.3562401007611,68.1422906748129,68.2062726166005,68.0498033914326,68.6018257745021,68.7606159741864]

plt.plot(xs,ys)
plt.show()


# Paste the contents of matplotlib_eg_scatter.py in the cell below and execute it.

# In[3]:

import matplotlib.pyplot as plt

xs = [0,82.0442626658045,164.140532825633,249.101916493353,332.182520454561,416.254670963008,495.330844802651,575.152803567278,653.205144040749,738.002602107511,818.027625347191,898.118246791891,980.079475188722,1061.14169245002,1139.27372549306,1222.46124841828,1300.35875796241,1375.56916894815,1450.58381247379,1528.61926864683,1605.58397275636,1690.73526419816,1765.8607155748,1844.76374052086,1927.68455634244,2007.86892722346,2086.83606915306,2166.71913598734,2251.80135023691,2335.70816516882,2420.79263834332,2499.86354911873,2577.05890165442,2654.19208140119,2730.09283946298,2809.07577275313,2894.08624312766,2969.05670691318,3049.93958564477,3129.75861740531,3206.68277248806,3283.60346903165,3364.58552562287,3449.40362446521,3525.4406380069,3608.13685182861,3690.05929028829,3773.14656467608,3850.22786804579,3930.30737684189,4015.46434790825,4093.50879464788,4172.64841837878,4252.66450124259,4335.57956650011,4420.52164467734,4495.52991250898,4575.39742270308,4650.49642074232,4730.52574677055,4805.68851119975,4880.60019974235,4960.70865715944,5043.67168991259,5118.73724644963,5203.54753389807,5283.38430140739,5359.51383129058,5438.3504224767,5519.19106748314,5597.38879699402]
ys = [69,70.5331166725557,71.5580608629997,71.7925736363559,71.3468902075964,70.8280365103217,70.6414982437025,70.4451053764375,70.3795952412711,69.7645648576213,69.5426154537094,69.3004779067882,68.9026747371149,68.599154906399,68.519069857217,68.3772433121902,69.1486631584102,68.7548264957532,68.3609585242701,68.5575021969504,68.5786288310157,70.1801398605569,69.7914710297875,70.1491959166127,71.3308639026079,71.9361568322099,71.7841530239482,71.5632100706621,70.9532872882161,70.4120650812618,69.7956890967353,69.6193830861858,69.6154149236862,69.6130427165012,69.7102463368457,69.5264209638887,68.9050613087613,69.1051712486311,68.8193037564598,68.6066819000086,68.6092749335593,68.6204785117934,68.3360360051156,68.5812312774717,68.3713770000006,69.5440095449469,70.5260768620176,72.0285425812187,72.0349254824144,71.8158332177273,71.1892654017154,71.1169346409977,70.8948191898231,70.6895661675202,70.2298913978404,69.6255849641529,69.7340391269025,69.5070018796742,69.6617353391113,69.4627544694931,69.630774569752,69.8026157834872,69.5269217792645,69.0549816241028,69.1984712324886,68.3562401007611,68.1422906748129,68.2062726166005,68.0498033914326,68.6018257745021,68.7606159741864]

plt.scatter(xs,ys)
plt.show()


# We'll use pandas to input the data and manage it so we'll need to import that package.

# In[4]:

import pandas as pd


# we will be working with data in the form of pandas DataFrame data.  DataFrame data are often indicated with variable names starting with df.  I have also appended '_con' to the variable name to indicate we are working with construction data.  This is time series data for public (governmental) construction spending and private construction spending.

# The pandas .read_csv() method is very useful.  You'll possibly need to adjust the path to the file based on where the .csv file is in relation to your Jupyter default folder.  

# In[5]:

#df_con = pd.read_csv('D:\\TeachingMaterials\\BusinessAnalytics\\Visualization\\VizData\\construction.csv')
df_con = pd.read_csv('C:\Users\Jing\Desktop\Competing Through Business Analytics\Storytelling & Visualization\data\DataVisualization\construction.csv')


# Use the .head() function to look at the first five rows of the DataFrame.  Note that the data files must have column headings in the first row.  Inspect the column headings

# In[6]:

df_con.head()


# This next statement imports the matplotlib graphing package into Jupyter using the very popular, if not ubiquitous alias, plt.

# In[7]:

import matplotlib.pyplot as plt


# We can extract a column of data from a DataFrame by using its name, as follows.  In this case the data are from the construction data set.

# In[8]:

df_con['Total Construction']


# Now, let's adapt the plotting code above to the df_con DataFrame data...  Note that Data Series are extracted from the pandas DataFrame using two different methods in the first two lines of the code.  If the column heading does not have any spaces, then you can use the method reflected in the first line.

# In[19]:

x = df_con['Month']
y = df_con['Total Construction']
y1 = df_con['Private Construction']
plt.plot(x,y,label='Total Construction') # The label parameter is a label for the y axis data that will be used in the legend   
plt.plot(x,y1,label='Private Construction')
plt.xlabel('Month')                      # Title for the horizontal axis
plt.ylabel('Construction Spending')      # Title for the vertical axis
plt.axis([x.min(),x.max(),0,1.1*y.max()])
plt.legend()
plt.savefig('sample.jpg')


# In[3]:

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
df_apple = pd.read_csv('C:\Users\Jing\Desktop\Competing Through Business Analytics\Storytelling & Visualization\data\correlate-apple_stock_price_mod.csv')

x = df_apple.index
y = df_apple['apple stock price']
y1 = df_apple['apple premarket']
plt.figure(figsize=(6,4))

plt.plot(x, y, label = 'apple stock price')
plt.plot(x, y1, label = 'apple premarket')
plt.xlabel('index')
plt.ylabel('standardized price')
plt.axis([x.min(),x.max()+100,0,1.1*y.max()])
plt.xticks(fontsize = 14)
plt.legend()


# In[4]:

x = df_apple['apple stock price']
y = df_apple['apple premarket']
plt.scatter(x, y, label = 'apple stock price vs. apple premarket')
plt.xlabel('apple stock price')
plt.ylabel('apple premarket')
plt.axis([x.min(),x.max(),0,1.1*y.max()])
plt.legend()


# In[24]:

# Extract data for graphing
x = df_con.Month[0:12]
y1 = df_con['Total Construction'][0:12]
y2 = df_con['Private Construction'][0:12]

# My variables
x_labels = ['','J','F','M','A','M','J','J','A','S','O','N','D']

# Create graph and assign Figure and Axes objects to variables fig and ax 
fig, ax = plt.subplots()

# Plot the data and set other Axes attributes
ax.plot(x,y1,label='Total Construction')                                  # Add y1 data to graph and create label for the legend
ax.plot(x,y2,label='Private Construction')                                # Add y2 data to graph and create label for the legend
ax.spines['right'].set_visible(False)                                                 # Remove right spine
ax.spines['top'].set_visible(False)                                                   # Remove top spine
ax.legend(loc = 'lower center', prop = {'family':'Times New Roman', 'size':'large'})  # Add legend and format it
ax.set_xlim(0,x.max()+1)                                                              # Set min and max for y axis
ax.set_ylim(0,1.1*y1.max())                                                           # Set min and max for y axis

# Set x-axis attributes
ax.xaxis.set_label_text('Month',fontsize = 18, fontname = 'Times New Roman')
ax.xaxis.set_ticks(range(0,13))
ax.xaxis.set_ticklabels(x_labels)
ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn tick marks off for top x axis

# Set y-axis attributes: the parameter 'both' refers to both major and minor tick marks
ax.yaxis.set_label_text('Construction Spending',fontsize = 18, fontname = 'Times New Roman')      # Title for the vertical axis
ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn tick marks off for right y axis

# Set Figure attributes
fig.set_size_inches(7,5)             # Set size of figure
fig.suptitle('Private vs. Total Construction',fontsize='xx-large', fontname = 'Times New Roman')
fig.tight_layout()                   # Helps with formatting and fitting everything into the figure
plt.savefig('sample3.jpg')           # Save jpg of figure


# The code below replicated the code above in many ways except that the two pandas Series are plotted on different axes within the figure

# In[20]:

# Extract data for graphing
x = df_con.Month[0:12]
y1 = df_con['Total Construction'][0:12]
y2 = df_con['Private Construction'][0:12]

# My variables
x_labels = ['','J','F','M','A','M','J','J','A','S','O','N','D']

# Create graph and assign Figure and Axes objects to variables fig and Axes variables ax1 and ax2
fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)  # Create 2 Axes (subplots) and assign to ax1 and ax2 respectively

# Plot the data and set other Axes attributes
ax1.plot(x,y1,label='Total Construction')                                  # Add y1 data to graph and create label for the legend
ax2.plot(x,y2,label='Private Construction')                                # Add y2 data to graph and create label for the legend

# Set common axes attributes
for ax in fig.axes:
    ax.spines['right'].set_visible(False)                                                 # Remove right spine
    ax.spines['top'].set_visible(False)                                                   # Remove top spine
    ax.legend(loc = 'upper right', prop = {'family':'Times New Roman', 'size':'large'})   # Add legend and format it
    ax.set_xlim(0,x.max()+1)                                                              # Set min and max for y axis
    ax.set_ylim(0,1.1*y1.max())                                                           # Set min and max for y axis

# Set x-axis attributes
for ax in fig.axes:
    ax.xaxis.set_label_text('Month',fontsize = 18)
    ax.xaxis.set_ticks(range(0,13))
    ax.xaxis.set_ticklabels(x_labels)
    ax.xaxis.set_tick_params(which = 'both', top = 'off', bottom = 'on', labelbottom = 'on')  # Turn top x axis tick marks off 

# Set y-axis attributes: the parameter 'both' refers to both major and minor tick marks
for ax in fig.axes:
    ax.yaxis.set_label_text('Construction Spending',fontsize = 18, fontname = 'Times New Roman')    # Title for the vertical axis
    ax.yaxis.set_tick_params(which = 'both', right = 'off', left = 'on', labelleft = 'on')   # Turn right y axis tick marks off 

# Set Figure attributes
fig.set_size_inches(12,5)            # Set size of figure
fig.tight_layout()                   # Helps with formatting and fitting everything into the figure
plt.savefig('sample3.jpg')           # Save jpg of figure


# In[26]:

x = df_con.index
y = df_con['Total Construction']
plt.figure(figsize=(24,4))               # figure must be resized prior to .plot() statement
plt.plot(x,y,label='Total Construction') 
plt.legend()
plt.xlabel('Month',fontsize=14)                      # Title for the horizontal axis
plt.ylabel('Construction Spending',fontsize=14)      # Title for the vertical axis
plt.axis([0,max(df_con['Month'])+1,0,max(df_con['Total Construction'])*1.05],fontsize=14)    # Set ranges of axes
plt.xticks(fontsize = 14)
plt.show()

