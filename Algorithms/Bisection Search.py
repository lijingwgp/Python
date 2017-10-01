# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 17:10:14 2017

@author: Jing
"""

def bisection_search_kth_root(N,k): 
    low = 0 
    high = 100
    #fun = (N)^(1/k)
    error = 0.001
    
    while low <= high: 
        mid = low + (high - low) / 2.0 
        
        if abs(mid**k - N) <= error: 
            return mid 
        if mid**k > N: 
            high = mid
        else: 
            low = mid
    
    return False


print bisection_search_kth_root(2,2)