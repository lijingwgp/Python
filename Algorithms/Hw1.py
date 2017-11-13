# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 16:06:05 2017

@author: Jing
"""

### Linear Search
### user need to define an array, and a target element
### algorithm will find the target element
### algorithm will return the index of that element
### linear performance
def linearSearch (F,target): 
    for x in range(len(F)): 
        if (F[x]==target): 
            return x 
    return False

F = [1,5,7,9,23,25,7,4545,75767,324234,657657,3242]
linearSearch(F,3)
###########################################################################


### Linear Search 
### user define the number
### algorithm will determine the square root of that number
### compare results with sqrt function from math package
from math import sqrt
def linearSearch_sqrt(N): 
    epsilon = 0.001 
    x = 0.000 
    while x*x < N - epsilon: 
        x += epsilon 
    return x

print linearSearch_sqrt(15)
print sqrt(15)
###########################################################################



### Binary Search
### user define an array, and a target element, a lower bound and an upper bound
### algorithm will compute a initial guess 
### algorithm will compare that initial guess number with the target number
### algorithm will decide whether to update lower bound or upper bond accordingly
### algorithm will return number of times of comparison
### log n performance
def binarySearch(A, target): 
    low = 0 
    high = len(A) - 1 
    idx = False
    
    while low <= high and not idx: 
        mid = low + (high - low) / 2 
        print "LOW {} HI {} MID {}, comparing {} to {}".format(low,high,mid,target,A[mid]) 
        
        if A[mid] == target: 
            return mid 
        if A[mid] > target: high = mid - 1 
        else: low = mid + 1 
    
    return False

F = range(32)
target = 4
print "Looking for [{}] in array {}".format(target,F)
print binarySearch(F, target)
############################################################################



### Bisection Search
### user define a number and kth root, a lower bound, an upper bound, and a tollerance
### algorithm will compute a initial guess
### algorithm will compute the absolute value of difference between the initial guess and 
### the user defined number
### if the difference is small enough to accept
### then the algorithm has find the kth root of that number
### algorithm will return that root
### log n performance
def bisection_search_kth_root(N,k): 
# N is the size of our problem
# k is the power
    low = 0                                     #lower bound
    high = 100000                               #upper bound
    error = 0.001                               #tolerance
    
    while low <= high: 
        guess = low + (high - low) / 2.0        #inital guess
        
        if abs(guess**k - N) <= error:          #best case
            return guess
        if guess**k > N:                        #average case 1
            high = guess
        else:
            low = guess                         #average case 2
    
    return False

print bisection_search_kth_root(100,2)
############################################################################



### Argmax
### user define a constraint
### algorithm will do a inital estimate of how large of a N! can ran on a 1TB memory
### algorithm will do a second more accurate run on the actual size of N! by using bisection search
### n log n performance
from math import log
def bisection_search_lgN(N):
# N is the constraint we defined     
    low = 0.0
    high = 0.001       
    error = 1

    while True:
        if high*log(high,2)-high+1 < N:                 # if the size of high smaller than our physical storage
            high = high * 2                             # we keep doubling
        else:                                           # this while loop is a first estimate of how large of computation we can made
            break

    while low <= high:                                  # the second estimate is more accurate    
        guess = low + (high - low) / 2.0                # initial guess  
        if abs((guess*log(guess,2)-guess+1) - N) <= error:      # when the difference between our estimation and actual constraint is acceptable
            return (guess, high) 
        if guess*log(guess,2)-guess+1 < N:                      # if our guess is smaller than constraint, update the lower bound
            low = guess    
        else:                                                   # if our guess is greater than constraint, update the upper bound
            high = guess
    return False

print bisection_search_lgN(2**43)
