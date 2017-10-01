# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:10:14 2017

@author: Jing
"""

def bisection_search_kth_root(N,k): 
    low = 0                                     #lower bound
    high = 100                                  #upper bound
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


print bisection_search_kth_root(2,2)