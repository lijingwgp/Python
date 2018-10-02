# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:35:44 2018

@author: jing.o.li
"""

############################
### Comparison Operators ###
############################

# write code to see if True equals False
True == False
# check if -5 * 15 is not equal to 75
-5*15 != 75
# comparison of strings
'pyscript' == 'PyScript'
# compare a boolean with an integer
True == 1
# Comparison of integers
x = -3*6
x >= -10
# Comparison of strings
y = "test"
"test" <= y
# Comparison of booleans
True > False

# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than or equal to 18
print(my_house >= 18)
# my_house less than your_house
print(my_house < your_house)



#########################
### Boolean Operators ###
#########################

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0
# my_kitchen bigger than 10 and smaller than 18?
# my_kitchen smaller than 14 or bigger than 17?
my_kitchen > 10 and my_kitchen < 18
my_kitchen < 14 or my_kitchen > 17

# Boolean operators with Numpy
# Before, the operational operators like < and >= worked with Numpy arrays out of the box. 
# Unfortunately, this is not true for the boolean operators and, or, and not.

import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])
# my_house greater than 18.5 or smaller than 10
np.logical_or(my_house>18.5, my_house<10)
# Both my_house and your_house smaller than 11
np.logical_and(my_house<11, your_house<11)



########################
#### if, elif, else ####
########################

z = 6
if z%2 == 0:
    print("z is even")
elif z%3 == 0:
    print("z is divisible by 3")
else:
    print("z is odd")



####################################
#### Filtering Pandas DataFrame ####
####################################

# 3 steps:
    # select the specified column
    # do comparison on the specified column
    # use this result to select other column

panda_series = data["column name"]
# or
panda_series = data.loc[:,"column name"]
panda_series = data.iloc[:,"column index"]
# now we are ready to do the comparison
is_huge = data["column name"] > 8
# pass this condition result to the panda data frame
data[is_huge]
# alternatively, we could use numpy array to solve this
data[np.logical_and(data["column name"]>8, data["column name"]<10)]
