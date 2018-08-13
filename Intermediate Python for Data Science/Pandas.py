# Pandas is an open source library, providing high-performance, 
# easy-to-use data structures and data analysis tools for Python. 
#
# The DataFrame is one of Pandas' most important data structures. 
# It's basically a way to store tabular data where you can label the rows and the columns. 
# One way to build a DataFrame is from a dictionary.
#
# Each dictionary key is a column label and each value is a list which contains the column elements.


# Pre-defined lists
import pandas as pd
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Create dictionary my_dict with three key:value pairs: my_dict
temp = ['country','drives_right','cars_per_cap']
my_dict = {temp[0]:names,temp[1]:dr,temp[2]:cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index=row_labels


# Putting data in a dictionary and then building a DataFrame works, but it's not very efficient. 
# What if you're dealing with millions of observations? In those cases, the data is typically 
# available as files with a regular structure. One of those file types is the CSV file


# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')
# Fix import by including index_col
cars = pd.read_csv('cars.csv', index_col = 0)
# Print out country column as Pandas Series
print(cars['country'])
# Print out country column as Pandas DataFrame
print(cars[['country']])
# Print out DataFrame with country and drives_right columns
print(cars[['country','drives_right']])
# Print out first 3 observations
print(cars[:3])


# With loc and iloc you can do practically any data selection operation on 
# DataFrames you can think of. loc is label-based, which means that you have to specify 
# rows and columns based on their row and column labels. iloc is integer index based, 
# so you have to specify rows and columns by their integer index 


# Print out observation for Japan
cars.loc[['JAP']]
# Print out observations for Australia and Egypt
cars.iloc[[1,6]]
# Print out drives_right value of Morocco
cars.loc['MOR', 'drives_right']
# Print out observations for Morocco with drives_right value
cars.loc[['MOR'], 'drives_right']
# Print sub-DataFrame
cars.loc[["MOR","RU"],["country","drives_right"]]
# Print out drives_right column as Series
cars.loc[:,'drives_right']
# Print out drives_right column as DataFrame
cars.iloc[:, [2]]
# Print out cars_per_cap and drives_right as DataFrame
cars.loc[:, ['cars_per_cap', 'drives_right']]
