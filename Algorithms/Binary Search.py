# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:52:50 2017

@author: Jing
"""

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