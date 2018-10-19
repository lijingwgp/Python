# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:17:55 2018

@author: jing.o.li
"""

##################################################
#### Bayesian Model-Based Optimization Primer ####
##################################################

# There are four parts to an optmization problem:
#       1. Objective function: what we want to minimize
#       2. Domain space: values of the parameters over which to minimize the objective
#       3. Hyperparameter optimization function: constructs the surrogate function and 
#          chooses next values to evaluate
#       4. Trials: score, parameter pairs recorded each time we evaluate the objective function

# The basic idea is that rather than just selecting from a grid uninformed by past objective 
# function evaluations, Bayesian methods take into account the previous results to try more 
# promising values. 
#
# This way, we limits calls to evaluate the objective function.
#
# A surrogate function which is a probability model of the objective function is much easier to
# optimize than the actual objective function.

# After each evaluation of the objective function, the algorithm updates the probability 
# model incorporating the new results.
# 
# SMBO methods are a formalization of Bayesian optimization that update the probability model
# sequentially: every evaluation of the objective function with a set of values updates the
# model with the idea that eventually the model will come to represent the true objective
# function. 
#
# This process is called Bayesian Reasoning: The algorithm forms an initial idea of the 
# objective function and updates it with each new piece of evidence.

# The next values to try in the objective function are selected by the algorithm optimizing the
# probability model (surrogate function) usually with a criteria known as Expected Improvement.
#
# Finding the values that will yield the greatest expected improvement in the surrogate 
# function is much cheaper than evaluating the objective function itself.
# 
# By choosing the next values based on a model rather than randomly, the hope is that 
# the algorithm with converge to the true best values much quicker.
#
# The overall goal is to evaluate the objective function fewer times by spending a little
# more time choosing the next values. 

# SMBO methods differ in part 3, the algorithm used to construct the probability.
# Several options are for the surrogate function are:
#       Gaussian Processes
#       Tree-structured Parzen Estimator
#       Random Forest Regression




##################
#### Hyperopt ####
##################

# Hyperopt is an open-source Python library for Bayesian optimization that implements SMBO
# using the Tree-structured Parzen Estimator. There are a number of libraries available 
# for Bayesian optimization and Hyperopt differs in that it is the only one to currently 
# offer the Tree Parzen Estimator. Other libraries use a Gaussian Process or a Random 
# Forest regression for the surrogate function (probability model).

# In this notebook, we will implement both random search (Hyperopt has a method for this) 
# as well as the Tree Parzen Estimator, a Sequential Model-Based Optimization method. 
# We will use a simple problem that will allow us to learn the basics as well as a number 
# of techniques that will be very helpful when we get to more complex use cases.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import hp




#####################
#### Preparation ####
#####################

### Objective
# For our objective function, we will use a simple polynomial function with the goal 
# being to find the minimu value. This function has one global minimum over the range
# we define it as well as one local minimum.

def objective(x):
    # Create the polynomial object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])
    # Return the value of the polynomial
    return f(x) * 0.05

# Space over which to evluate the function is -5 to 6
x = np.linspace(-5, 6, 10000)
y = objective(x)

# Visualize the function
miny = min(y)
minx = x[np.argmin(y)]
plt.figure(figsize = (8, 6))
plt.style.use('fivethirtyeight')
plt.title('Objective Function'); plt.xlabel('x'); plt.ylabel('f(x)')
plt.vlines(minx, min(y)- 50, max(y), linestyles = '--', colors = 'r')
plt.plot(x, y)
# Print out the minimum of the function and value
print('Minimum of %0.4f occurs at %0.4f' % (miny, minx))


### Domain
# The domain is the values of x over which we evaluate the function. First we can use a 
# uniform distribution over the space our function is defined.

# Create the domain space
space = hp.uniform('x', -5, 6)

# We can draw samples from the space using a Hyperopt utility. This is useful for
# visualizing a distribution
from hyperopt.pyll.stochastic import sample
samples = []

# Sample 10000 values from the range
for _ in range(10000):
    samples.append(sample(space))
    
# Histogram of the values
plt.hist(samples, bins = 20, edgecolor = 'black'); 
plt.xlabel('x');plt.ylabel('Frequency');plt.title('Domain Space')

# Later, our algorithm will sample values from this distribution, initially at random 
# as it explores the domain space, but then over time, it will "focus" on the most promising
# values. 

# Therefore, the algorithm should more values around 4.9, the minimum of the function. 
# We can compare this to random search which should try values evenly from the entire 
# distribution.


### Hyperparameter Optimization Algorithm
# There are two choices for a hyperparameter optimization algorithm in Hyperopt: random
# and Tree Parzen Estimator. We could use both and compare the results
from hyperopt import rand, tpe

# Create the algorithms
tpe_algo = tpe.suggest
rand_algo = rand.suggest


### History
# Storing the history is as simple as making a Trails object that we pass into the function call.
from hyperopt import Trials

# Create two trials objects
tpe_history = Trials()
rand_history = Trials()




##############################
#### Run the Optimization ####
##############################

# Now that all four parts are in place, we are ready to minimize.
# Let's run 2000 iterations of the minimization with both the random algorithm
# and the Tree Parzen Estimator algorithm

# The fimin ufnction takes in exactly the four parts specified above as well as the maximum number
# of evaluations to run. We will also set a rstate for reproducible results across multiple runs.

from hyperopt import fmin

# Run 2000 evals with the tpe algorithm
tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_history,
                max_evals=2000, rstate=np.random.RandomState(123))
# Run 2000 evals with the random algorithm
rand_best = fmin(fn=objective, space=space, algo=rand_algo, trials=rand_history,
                max_evals=2000, rstate=np.random.RandomState(123))

# Print out information about losses
print('Minimum loss attained with TPE:    {:.4f}'.format(tpe_history.best_trial['result']['loss']))
print('Minimum loss attained with random: {:.4f}'.format(rand_history.best_trial['result']['loss']))
print('Actual minimum of f(x):            {:.4f}'.format(miny))

# Print out information about value of x
print('\nBest value of x from TPE:  {:.4f}'.format(tpe_best['x']))
print('Best value of x from random: {:.4f}'.format(rand_best['x']))
print('Actual best value of x:      {:.4f}'.format(minx))

# Print out information about number of trials
print('\nNumber of trials needed to attain minimum with TPE:  {}'.format(tpe_history.best_trial['misc']['idxs']['x'][0]))
print('Number of trials needed to attain minimum with random: {}'.format(rand_history.best_trial['misc']['idxs']['x'][0]))

%%timeit -n 3
best = fmin(fn=objective, space=space, algo=tpe_algo, max_evals=200)
%%timeit -n 3
best = fmin(fn=objective, space=space, algo=rand_algo, max_evals=200)

# As a point interest, the random algorithm ran about 5 times faster than the tpe 
# algorithm. This shows that the TPE method is taking more time to propose the next set 
# of parameters while the random method is just choosing from the space well, randomly. 
# The extra time to choose the next parmaeters is made up for by choosing better 
# parameters that should let us make fewer overall calls to the objective function 
# (which is the most expensive part of optimization).




#####################
#### TPE Results ####
#####################

# We see that both models returned values very close to the optimal. 
tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_history.results], 
                            'iteration': tpe_history.idxs_vals[0]['x'],
                            'x': tpe_history.idxs_vals[1]['x']})
plt.figure(figsize = (10, 8))
plt.plot(tpe_results['iteration'], tpe_results['x'], 'bo', alpha = 0.5);
plt.xlabel('Iteration', size = 22); plt.ylabel('x value', size = 22); 
plt.title('TPE Sequence of Values', size = 24);
plt.hlines(minx, 0, 2000, linestyles = '--', colors = 'r');

# We can see that over time, the algorithm tended to try values closer to 4.9. 
# The local minimum around -4 likely threw off the algorithm initially, but the points 
# tend to cluster around the actual minimum as the algorithm progresses.

# We can also plot the histogram to see the distribution of values tried.
plt.figure(figsize = (8, 6))
plt.hist(tpe_results['x'], bins = 50, edgecolor = 'k');
plt.title('Histogram of TPE Values'); plt.xlabel('Value of x'); plt.ylabel('Count');




########################
#### Random Results ####
########################

rand_results = pd.DataFrame({'loss': [x['loss'] for x in rand_history.results], 'iteration': rand_history.idxs_vals[0]['x'],
                            'x': rand_history.idxs_vals[1]['x']})
plt.figure(figsize = (10, 8))
plt.plot(rand_results['iteration'], rand_results['x'],  'bo', alpha = 0.5);
plt.xlabel('Iteration', size = 22); plt.ylabel('x value', size = 22); plt.title('Random Sequence of Values', size = 24);
plt.hlines(minx, 0, 2000, linestyles = '--', colors = 'r');

# Sort with best loss first
rand_results = rand_results.sort_values('loss', ascending = True).reset_index()
plt.figure(figsize = (8, 6))
plt.hist(rand_results['x'], bins = 50, edgecolor = 'k');
plt.title('Histogram of Random Values'); plt.xlabel('Value of x'); plt.ylabel('Count');




####################################
#### Slightly Advanced Concepts ####
####################################

# There are smarter concepts because they will make our jobs easier.
#       smarter domain space over which to search
#       return more useful information from the objective function

# Better Domain Space
# In more complicated problems, we don't have a graph to show us the minimum, 
# but we can still use experience and knowledge to inform our choice of a domain space.

# Here we will make a normally distributed domain space around the value where the minimum
# of the objective function occurs, around 4.9.

samples = []
space = hp.normal('x', 4.9, 0.5)
for _ in range(10000):
    samples.append(sample(space))
plt.hist(samples, bins = 20, edgecolor = 'black'); 
plt.xlabel('x'); plt.ylabel('Frequency'); plt.title('Domain Space');

# More useful trials object
# Another modification to make is to return more useful information from the objective 
# function. We do this using a dictionary with any information we want included.
# 
# The only requirements are that the dictionary must contain a single real-valued metric
# to minimize stored under a "loss" key and whether the function sucessfully ran, stored
# under a "status" key. Here we make the modifications to the objective to store the values
# of x as well as the time to evaluate

from hyperopt import STATUS_OK
from timeit import default_timer as timer

def objective(x):    
    # Create the polynomial object
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])
    # Evaluate the function
    start = timer()
    loss = f(x) * 0.05
    end = timer()
    # Calculate time to evaluate
    time_elapsed = end - start
    results = {'loss': loss, 'status': STATUS_OK, 'x': x, 'time': time_elapsed}
    # Return dictionary
    return results

# New trials object
tpe_history = Trials()
# Run 2000 evals with the tpe algorithm
best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_history, 
                max_evals=2000, rstate= np.random.RandomState(120))

# This time our trials object will have all of our information in the results attribute
results = tpe_history.results
results[:2]
# Results into a dataframe
results_df = pd.DataFrame({'time': [x['time'] for x in results], 
                           'loss': [x['loss'] for x in results],
                           'x': [x['x'] for x in results],
                            'iteration': list(range(len(results)))})
# Sort with lowest loss on top
results_df = results_df.sort_values('loss', ascending = True)
results_df.head()

# Histogram of optimized x values
plt.hist(results_df['x'], bins = 50, edgecolor = 'k');
plt.title('Histogram of TPE Values'); plt.xlabel('Value of x'); plt.ylabel('Count');
# Comparison between the two distribution
sns.kdeplot(results_df['x'], label = 'Normal Domain')
sns.kdeplot(tpe_results['x'], label = 'Uniform Domain')
plt.legend(); plt.xlabel('Value of x'); plt.ylabel('Density'); plt.title('Comparison of Domain Choice using TPE');