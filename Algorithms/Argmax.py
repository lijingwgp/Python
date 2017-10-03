# -*- coding: utf-8 -*-
"""
Created on Sun Oct 01 17:02:49 2017

@author: Jing
"""

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
        
        


      
