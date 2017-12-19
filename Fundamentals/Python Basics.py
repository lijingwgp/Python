# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:50:18 2017

@author: Jing
"""

## whitespace is important
listOfNumbers = [1, 2, 3, 4, 5, 6]
for number in listOfNumbers:
    print(number)
    if (number % 2 == 0):
        print("is even")
    else:
        print("is odd")
        
print ("All done.")


## importing modules
import numpy as np
A = np.random.normal(25.0, 5.0, 10)
print (A)


## lists
x = [1, 2, 3, 4, 5, 6]
print(len(x))
x[:3]
x[3:]
x[-2:]
x.extend([7,8])
x.append(9)
x
y = [10, 11, 12]
listOfLists = [x, y]
listOfLists
z = [3, 2, 1]
z.sort()
z.sort(reverse=True)
z


## tuples
## just like lists, but immutable, use () instead of []
x = (1, 2, 3)
len(x)
y = (4, 5, 6)
y[2]
listOfTuples = [x, y]
listOfTuples
(age, income) = "32,120000".split(',')
print(age)
print(income)


## dictionaries
## like a map 
captains = {}
captains["Enterprise"] = "Kirk"
captains["Enterprise D"] = "Picard"
captains["Deep Space Nine"] = "Sisko"
captains["Voyager"] = "Janeway"
print(captains["Voyager"])
print(captains.get("NX-01"))
for ship in captains:
    print(ship + ": " + captains[ship])
    

## functions
def SquareIt(x):
    return x * x
print(SquareIt(2))
## you can pass functions around as parameters
def DoSomething(f, x):
    return f(x)
print(DoSomething(SquareIt, 3))
## inline simple functions
print(DoSomething(lambda x: x * x * x, 3))


## boolean expressions
print(1 == 3)
print(True or False)
print(1 is 3)
if 1 is 3:
    print("How did that happen?")
elif 1 > 3:
    print("Yikes")
else:
    print("All is well with the world")
    

## looping
for x in range(10):
    print(x)
for x in range(10):
    if (x is 1):
        continue
    if (x > 5):
        break
    print(x)
x = 0
while (x < 10):
    print(x)
    x += 1
