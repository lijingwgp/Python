# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:16:17 2018

@author: jing.o.li
"""

import csv
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import STATUS_OK
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer

data = pd.read_csv('caravan-insurance-challenge.csv')
train = data[data['ORIGIN'] == 'train']
test = data[data['ORIGIN'] == 'test']
train_labels = np.array(train['CARAVAN'].astype(np.int32)).reshape((-1,))
test_labels = np.array(test['CARAVAN'].astype(np.int32)).reshape((-1,))
train = train.drop(columns = ['ORIGIN', 'CARAVAN'])
test = test.drop(columns = ['ORIGIN', 'CARAVAN'])
train_features = np.array(train)
test_features = np.array(test)
train_labels = train_labels[:]
train_set = lgb.Dataset(train_features, label = train_labels)
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)



#############################################################
#### Bayesian Hyperparameter Optimization using Hyperopt ####
#############################################################

# For Bayesian optimization, we need the following four parts:
#       1. Objective function
#       2. Domain space
#       3. Hyperparameter optimization algorithm
#       4. History of results

### Objective Function
# The only requirement for an objective function in Hyperopt is that it has a key in 
# the return dictionary called "loss" to minimize and a key called "status" indicating 
# if the evaluation was successful.

# If we want to keep track of the number of iterations, we can declare a global variables 
# called ITERATION that is incremented every time the function is called. In addition to 
# returning comprehensive results, every time the function is evaluated, we will write 
# the results to a new line of a csv file.

# The most important part of this function is that now we need to return a value to 
# minimize and not the raw ROC AUC. We are trying to find the best value of the objective
# function, and even though a higher ROC AUC is better, Hyperopt works to minimize a 
# function. Therefore, a simple solution is to return 1 - ROC

ITERATION = 0
def objective(params, n_folds = 5):
    """Objective function for LightGBM Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    ITERATION += 1
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    run_time = timer() - start
    
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}

### Domain Space
# In Hyperopt, and other Bayesian optimization frameworks, the domian is not a discrete
# grid but instead has probability distributions for each hyperparameter. For each 
# hyperparameter, we will use the same limits as with the grid, but instead of being 
# defined at each point, the domain represents probabilities for each hyperparameter. 

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

# we are using a log-uniform space for the learning rate defined from 0.005 to 0.2

learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
learning_rate_dist = []
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])
    
plt.figure(figsize = (8, 6))
sns.kdeplot(learning_rate_dist, color = 'red', linewidth = 2, shade = True);
plt.title('Learning Rate Distribution', size = 18); 
plt.xlabel('Learning Rate', size = 16); plt.ylabel('Density', size = 16);

# The number of leaves is again a uniform distribution. Here we used quniform 
# which means a discrete uniform
# Discrete uniform distribution

num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
num_leaves_dist = []
for _ in range(10000):
    num_leaves_dist.append(sample(num_leaves)['num_leaves'])
plt.figure(figsize = (8, 6))
sns.kdeplot(num_leaves_dist, linewidth = 2, shade = True);
plt.title('Number of Leaves Distribution', size = 18); plt.xlabel('Number of Leaves', size = 16); plt.ylabel('Density', size = 16);

# Now, below is the complete Bayesian Domain

space = {
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

### Optimization Algorithm
# Although this is the most technical part of Bayesian optimization, defining the 
# algorithm to use in Hyperopt is simple. We will use the Tree Parzen Estimator
# which is one method for constructing the surrogate function and choosing the next 
# hyperparameters to evaluate.

from hyperopt import tpe
tpe_algo = tpe.suggest

### Result History
# The final part is the result history. Here, we are using two methods to make sure 
# we capture all the results:
#       1. A Trials object that stores the dictionary returned
#       2. Writing to a csv file every iteration

from hyperopt import Trials
bayes_trials = Trials()

# The Trials object will hold everything returned from the objective function in the 
# .results attribute. It also holds other information from the search.

# File to save first results
out_file = 'lightgbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()



###############################
#### Bayesian Optimization ####
###############################

# Now, we have everything in place needed to run the optimization. First, we declare
# a global variable that will be used to keep track of the number of iterations. 
# Then, we call fmin and passing in everything we defined above and the maximum number
# iterations to run.

from hyperopt import fmin
max_iters = 500

best = fmin(fn = objective, space = space, algo = tpe_algo, max_evals = max_iters,
            trials = bayes_trials, rstate = np.random.RandomState(50))

# The .results attribute of the Trials object has all information from the objective 
# function. If we sort this by the lowest loss, we can see the hyperparameters that 
# performed the best in terms of validation loss.

bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
bayes_trials_results[:2]

# We can also access the results from the csv file
results = pd.read_csv('lightgbm_trials.csv')
# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()

# For some reason, when we save to a file and then read back in, the dictionary of 
# hyperparameters is represented as a string. To convert from a string back to a 
# dictionary we can use the ast library and the literal_eval function.

import ast
ast.literal_eval(results.loc[0, 'params'])



###############################
#### Evaluate Best Results ####
###############################

# Now for the moment of truth: did the optimization pay off?
# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs = -1, 
                                       objective = 'binary', random_state = 50, **best_bayes_params)
best_bayes_model.fit(train_features, train_labels)

# Evaluate on the testing data 
preds = best_bayes_model.predict_proba(test_features)[:, 1]
print('The best model from Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(test_labels, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))

