# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 21:43:12 2018

@author: Jing
"""

from mrjob.job import MRJob
from mrjob.step import MRStep


class MRsales(MRJob):
    def ConvertCurrency(self, us_dollar):
        canada_dollar = float(us_dollar) * 1.24 * 0.7
        return canada_dollar

    def steps(self):
        return [
            MRStep(mapper = self.mapper_get_currency,
                   combiner = self.combiner_avg_sales,
                   reducer = self.reducer_flip_keys),
            MRStep(reducer = self.reducer_find_max)
        ]
    
    def mapper_get_currency(self, _, line):
        (ProductID, Sales, Quantity, Profit) = line.split(',')
        money = self.ConvertCurrency(Sales)
        yield (ProductID, money)
    
    def combiner_avg_sales(self, ProductID, moneys):
        totalSales = 0
        numElements = 0
        for money in moneys:
            totalSales += money
            numElements += 1
        yield (ProductID, totalSales/numElements)
        
    def reducer_flip_keys(self, ProductID, moneys):
        totalSales = 0
        numElements = 0
        for money in moneys:
            totalSales += money
            numElements += 1
        yield None, (totalSales/numElements, ProductID)
      
    def reducer_find_max(self, key, sale_product_pairs):
        yield max(sale_product_pairs)
        
        
if __name__ == '__main__':
    MRsales.run()
#!python MostSales.py ProductID.csv > sales.txt