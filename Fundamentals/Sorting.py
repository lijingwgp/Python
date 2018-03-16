# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:47:53 2018

@author: Jing
"""

# sorting a list with numerical values
# create a list
a = [1,2,3,4,5,34,23,12]
# sort accendingly
a.sort()
a
# sort accendingly but not save
print sorted(a)
a
# sort decendingly
a.sort(reverse=True)
a


# sorting a list of tuples
# create a list of tuples
a = [(1,2), (7,3), (10,4), (3,5)]
# sort this list
a.sort()
a
# notice this is sorting the list according to the first element of each tuple
# the same goes with list of lists
#
# now, sort according to the second element of each tuple
import operator
a.sort(key=operator.itemgetter(1))
a
# sort based on the second element of each tuple but in reverse order
a.sort(key=operator.itemgetter(1), reverse=True)
a


# sorting a list with lambda functions
# define a lambda function
g = lambda x:x**2
g(3)
# define a lambda function that does sorting
# this lambda function that returns the second element of the tuple 
h = lambda x:x[1]
h((1,5,10))
h([2,4,19,3])
# sort using the lambda function
a = [(1,2,3),(4,12,23),(45,100,0)]
a.sort(key=lambda x:x[1])
a
# reverse order
a.sort(key=lambda x:x[1], reverse=True)
a


# sorting a dictionary
knapsack = {0:(5,2), 1:(3,4), 2:(13,23)}
# iterating a dictionary
for k,v in knapsack.items():
    print k,v
# generate a list based on the keys and values of a dictionary
item = [[k,v] for k,v in knapsack.items()]
item
# now, item is a list of lists
# each sublist contains two things, an integer which is a key from the dictionary, and a tuple
# which is a value from the dictionary
item[0][1][0]
j = lambda x:x[1][0] 
j(item[0])
item.sort(key=j)
item










