# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 10:24:12 2018

@author: jing.o.li
"""

####################
#### While Loop ####
####################

# while loop = repeated if statements as long as the condition is true
error = 50.0
while error > 1:
    error = error/4
    print(error)



##################
#### For Loop ####
##################
    
fam = [1.73,2.21,3.78,4.06,5.55]
for height in fam:
    print(height)

# or print element index along with each output
for index, height in enumerate(fam):
    print("index" + str(index) + ": " + str(height))

#
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
for each in house:
    print("the "+str(each[0])+" is "+str(each[1])+" sqm")



#################################
#### Looping Data Structures ####
#################################

# looping dictionaries
world = {"afghanistan": 30.44,
         "albania": 27.99,
         "algeria": 2.34}
# note that the order is wrong. this is why sorting is important
for key, value in world.items():
    print(key + " -- " + str(value))

# looping numpy arrays
height = np.array([1.34,2.33,5.78])
weight = np.array([45,39.1,899])
bmi = weight / height **2
for each in bmi:
    print(each)

# looping 2D numpy arrays
meas = np.array([height, weight])
for each in meas:
    print(each)
# to access each element 
for each in np.nditer(meas):
    print(each)

# looping pandas dataframe
import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
# this only outpus column names
for each in iris:
    print(each)
# this will output row and column items
for lab, row in iris.iterrows():
    print(lab)
    print(row)
# print specific item value based on row index
for lab, row in iris.iterrows():
    print(str(lab) + ": " + str(row["species"]))
# add column
for lab, row in iris.iterrows():
    iris.loc[lab,"species_length"] = len(row["species"])
# or
iris["species_length2"] = iris["species"].apply(len)
