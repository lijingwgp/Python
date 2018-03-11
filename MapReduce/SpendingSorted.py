# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:12:11 2018

@author: Jing
"""

from mrjob.job import MRJob
from mrjob.step import MRStep


class MRSpending_Sorted(MRJob):
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_amount,
                   reducer=self.reducer_sum_amount),
            MRStep(mapper=self.mapper_sorted_sum_amount,
                   reducer = self.reducer_output_sum_amount)
        ]
    
    def mapper_get_amount(self, _, line):
        (CustomerID, ItemID, Amount) = line.split(',')
        yield CustomerID, float(Amount)
        
    def reducer_sum_amount(self, CustomerID, Amount):
        yield CustomerID, sum(Amount)
    
    def mapper_sorted_sum_amount(self, CustomerID, Order):
        yield '%04.02f'%float(Order), CustomerID
    
    def reducer_output_sum_amount(self, Order, CustomerIDs):
        for CustomerID in CustomerIDs:
            yield Order, CustomerID
            
        
if __name__ == '__main__':
    MRSpending_Sorted.run()
# !python SpendingSorted.py DataA1.csv > spendamountsorted.txt
