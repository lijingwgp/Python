# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:29:01 2020

@author: 607991
"""

# TODO: import necessary libraries

import math
import matplotlib as plt
from .Generaldistribution import Distribution

# TODO: make a Binomial class that inherits from the Distribution class. Use the specifications below.

class Binomial(Distribution):
    
    """ Binomial distribution class for calculating and 
    visualizing a Binomial distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats to be extracted from the data file
        p (float) representing the probability of an event occurring
        n (int) number of trials
    """

    # TODO: define the init function

    def __init__(self, prob=.5, size=25):

        Distribution.__init__(self)
        self.p = prob
        self.n = size

    # TODO: write a method calculate_mean() according to the specifications below

    def calculate_mean(self):

        """Function to calculate the mean from p and n

        Args: 
            None

        Returns: 
            float: mean of the data set
        """ 

        return self.n * self.p

    #TODO: write a calculate_stdev() method accordin to the specifications below.

    def calculate_stdev(self):

        """Function to calculate the standard deviation from p and n.

        Args: 
            None

        Returns: 
            float: standard deviation of the data set
        """
        
        return math.sqrt(self.n * self.p * (1 - self.p))

    # TODO: write a replace_stats_with_data() method according to the specifications below. The read_data_file() 
    # from the Generaldistribution class can read in a data file. Because the Binomaildistribution class inherits 
    # from the Generaldistribution class, you don't need to re-write this method. However, that method
    # doesn't update the mean or standard deviation of a distribution. 
    # Hence you are going to write a method that calculates n, p, mean and
    # standard deviation from a data set and then updates the n, p, mean and stdev attributes.
    # Assume that the data is a list of zeros and ones like [0 1 0 1 1 0 1]. 
    #
    #       Write code that: 
    #           updates the n attribute of the binomial distribution
    #           updates the p value of the binomial distribution by 
    #           calculating the number of positive trials divided by the total trials
    #           updates the mean attribute
    #           updates the standard deviation attribute
    #
    #       Hint: You can use the calculate_mean() and calculate_stdev() methods
    #           defined previously.

    def replace_stats_with_data(self):

        """Function to calculate p and n from the data set. The function updates the p and n variables of the object.

        Args: 
            None

        Returns: 
            float: the p value
            float: the n value
        """
        
        self.p = 1.0*sum(self.data) / len(self.data)
        self.n = len(self.data)
        self.mean = self.calculate_mean()
        self.stdev = self.calculate_stdev()
        return self.p, self.n
    
    # TODO: write a method plot_bar() that outputs a bar chart of the data set according to the following specifications.
    
    def plot_bar(self):
    
        """Function to output a histogram of the instance variable data using 
        matplotlib pyplot library.
        
        Args:
            None
            
        Returns:
            None
        """
        
        plt.bar(x = ['0', '1'], height = [(1 - self.p) * self.n, self.p * self.n])
        plt.title('Bar Chart of Data')
        plt.xlabel('outcome')
        plt.ylabel('count')
    
    #TODO: Calculate the probability density function of the binomial distribution
    
    def pdf(self, k):
        
        """Probability density function calculator for the binomial distribution.
        
        Args:
            k (float): point for calculating the probability density function
           
        Returns:
            float: probability density function output
        """
        
        a = math.factorial(self.n) / (math.factorial(k) * (math.factorial(self.n - k)))
        b = (self.p ** k) * (1 - self.p) ** (self.n - k)
        return a * b

    # write a method to plot the probability density function of the binomial distribution
    
    def plot_bar_pdf(self):

        """Function to plot the pdf of the binomial distribution
        
        Args:
            None
        
        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot
        """
        
        x = []
        y = []
        
        # calculate the x values to visualize
        for i in range(self.n + 1):
            x.append(i)
            y.append(self.pdf(i))

        # make the plots
        plt.bar(x, y)
        plt.title('Distribution of Outcomes')
        plt.ylabel('Probability')
        plt.xlabel('Outcome')

        plt.show()

        return x, y
                
    # write a method to output the sum of two binomial distributions. Assume both distributions have the same p value.
    
    def __add__(self, other):
        
        """Function to add together two Binomial distributions with equal p
        
        Args:
            other (Binomial): Binomial instance
            
        Returns:
            Binomial: Binomial distribution
        """
        
        try:
            assert self.p == other.p, 'p values are not equal'
        except AssertionError as error:
            raise
            
        result = Binomial()
        result.n = self.n + other.n
        result.p = self.p
        result.mean = result.calculate_mean()
        result.stdev = result.calculate_stdev()
        return result
    
    # use the __repr__ magic method to output the characteristics of the binomial distribution object.
    
    def __repr__(self, other):
    
        """Function to output the characteristics of the Binomial instance
        
        Args:
            None
        
        Returns:
            string: characteristics of the Binomial object
        """
        
        return "mean {}, standard deviation {}, p {}, n {}".\
        format(self.mean, self.stdev, self.p, self.n)