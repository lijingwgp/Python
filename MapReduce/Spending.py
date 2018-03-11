# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:32:19 2018

@author: Jing
"""

from mrjob.job import MRJob


class MRSpending(MRJob):
    
    def mapper(self, _, line):
        (CustomerID, ItemID, Amount) = line.split(',')
        yield CustomerID, float(Amount)
    
    def reducer(self, CustomerID, Amount):
        yield CustomerID, sum(Amount)
    

if __name__ == '__main__':
    MRSpending.run()
# !python Spending.py DataA1.csv > spendamount.txt