# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:09:56 2018

@author: jing.o.li
"""

import ast
import csv
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin



##########################
#### Data Preparation ####
##########################

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



##################
#### Bayesian ####
##################

# Surrogate
ITERATION = 0
def objective(params, n_folds = 5):
    """Objective function for LightGBM Hyperparameter Optimization"""
    global ITERATION
    ITERATION += 1

    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    run_time = timer() - start

    best_score = np.max(cv_results['auc-mean'])
    loss = 1 - best_score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])
    
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}

### Domain
space = {
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

### Parameters Selection
tpe_algo = tpe.suggest

### Result History
bayes_trials = Trials()
out_file = 'lightgbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

### Run Bayesian Optimization
max_iters = 500
best = fmin(fn = objective, space = space, algo = tpe_algo, max_evals = max_iters,
            trials = bayes_trials, rstate = np.random.RandomState(50))

### Modeling Results
results = pd.read_csv('lightgbm_trials.csv')
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()



#######################
#### Random Search ####
#######################

### Data set
train_set = lgb.Dataset(train_features, label = train_labels)

### Domain
param_grid = {
    'boosting_type': ['gbdt'],
    'num_leaves': list(range(30, 80)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10))
}

### Modeling Results
random_results = pd.DataFrame(columns = ['loss', 'params', 'iteration', 'estimators', 'time'],
                       index = list(range(max_iters)))

### Objective Function
folds = 5
def random_objective(params, iteration, n_folds = folds):
    """Random search objective function. Takes in hyperparameters
       and returns a list of results to be saved."""
    start = timer()
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    end = timer()
    best_score = np.max(cv_results['auc-mean'])    
    loss = 1 - best_score    
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)    
    return [loss, params, iteration, n_estimators, end - start]

### Run Random Search
random.seed(50)
for i in range(max_iters):    
    params = {key: random.sample(value, 1)[0] for key, value in param_grid.items()}
    print(params)
    results_list = random_objective(params, i)    
    random_results.loc[i, :] = results_list

# Accessing the Results
random_results.sort_values('loss', ascending = True, inplace = True)
random_results.reset_index(inplace = True, drop = True)
random_results.head()
best_random_params = random_results.loc[0, 'params'].copy()
best_random_estimators = int(random_results.loc[0, 'estimators'])



#####################################
#### Comparison to Random Search ####
#####################################

best_random_params['method'] = 'Random Search'
best_bayes_params['method'] = 'Bayesian Optimization'
best_params = pd.DataFrame(best_bayes_params, index = [0]).append(pd.DataFrame(best_random_params, index = [0]), 
                                                                  ignore_index = True, sort = True)
best_params



#####################################
#### Visualizing Hyperparameters ####
#####################################

# First we can make a kernel density estimate plot of the learning_rate sampled in 
# random search and Bayes Optimization. As a reference, we can also show the sampling 
# distribution.

# Create a new dataframe for storing parameters
random_params = pd.DataFrame(columns = list(random_results.loc[0, 'params'].keys()),
                            index = list(range(len(random_results))))
# Add the results with each parameter a different column
for i, params in enumerate(random_results['params']):
    random_params.loc[i, :] = list(params.values())
random_params['loss'] = random_results['loss']
random_params['iteration'] = random_results['iteration']
random_params.head()

# Do the same for Bayesian tuned parameters
bayes_params = pd.DataFrame(columns = list(ast.literal_eval(results.loc[0, 'params']).keys()),
                            index = list(range(len(results))))
for i, params in enumerate(results['params']):
    bayes_params.loc[i, :] = list(ast.literal_eval(params).values())
bayes_params['loss'] = results['loss']
bayes_params['iteration'] = results['iteration']
bayes_params.head()

# Density plots of the learning rate distributions 
learning_rate_dist = []
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18
sns.kdeplot(learning_rate_dist, label = 'Sampling Distribution', linewidth = 2)
sns.kdeplot(random_params['learning_rate'], label = 'Random Search', linewidth = 2)
sns.kdeplot(bayes_params['learning_rate'], label = 'Bayes Optimization', linewidth = 2)
plt.legend()
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');

# Iterate through each hyperparameter
for i, hyper in enumerate(random_params.columns):
    if hyper not in ['class_weight', 'boosting_type', 'iteration', 'subsample', 'metric', 'verbose']:
        plt.figure(figsize = (14, 6))
        # Plot the random search distribution and the bayes search distribution
        if hyper != 'loss':
            sns.kdeplot([sample(space[hyper]) for _ in range(1000)], label = 'Sampling Distribution')
        sns.kdeplot(random_params[hyper], label = 'Random Search')
        sns.kdeplot(bayes_params[hyper], label = 'Bayes Optimization')
        plt.legend(loc = 1)
        plt.title('{} Distribution'.format(hyper))
        plt.xlabel('{}'.format(hyper)); plt.ylabel('Density');
        plt.show();



##############################
#### Evolution of Search #####
##############################

# As the optimization progresses, we expect the Bayes method to focus on the 
# more promising values of the hyperparameters: those that yield the lowest 
# error in cross validation. We can plot the values of the hyperparameters 
# versus the iteration to see if there are noticeable trends.

# Plot of four hyperparameters
fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0
for i, hyper in enumerate(['colsample_bytree', 'learning_rate', 'min_child_samples', 'num_leaves']):
        # Scatterplot
        sns.regplot('iteration', hyper, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));
plt.tight_layout()

# Scatterplot of next three hyperparameters
fig, axs = plt.subplots(1, 3, figsize = (18, 6))
i = 0
for i, hyper in enumerate(['reg_alpha', 'reg_lambda', 'subsample_for_bin']):
        sns.regplot('iteration', hyper, data = bayes_params, ax = axs[i])
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));
plt.tight_layout()

# Finally, we can look at the losses recorded by both random search and Bayes Optimization.
scores = pd.DataFrame({'ROC AUC': 1 - random_params['loss'], 'iteration': random_params['iteration'], 'search': 'random'})
scores = scores.append(pd.DataFrame({'ROC AUC': 1 - bayes_params['loss'], 'iteration': bayes_params['iteration'], 'search': 'Bayes'}))
scores['ROC AUC'] = scores['ROC AUC'].astype(np.float32)
scores['iteration'] = scores['iteration'].astype(np.int32)
scores.head()

# We can make histograms of the scores (not taking in account the iteration) 
# on the same x-axis scale to see if there is a difference in scores.

plt.figure(figsize = (18, 6))
# Random search scores
plt.subplot(1, 2, 1)
plt.hist(1 - random_results['loss'].astype(np.float64), label = 'Random Search', edgecolor = 'k');
plt.xlabel("Validation ROC AUC"); plt.ylabel("Count"); plt.title("Random Search Validation Scores")
plt.xlim(0.75, 0.78)
# Bayes optimization scores
plt.subplot(1, 2, 2)
plt.hist(1 - bayes_params['loss'], label = 'Bayes Optimization', edgecolor = 'k');
plt.xlabel("Validation ROC AUC"); plt.ylabel("Count"); plt.title("Bayes Optimization Validation Scores");
plt.xlim(0.75, 0.78);

# Bayesian optimization should get better over time. Let's plot the scores 
# against the iteration to see if there was improvement.

sns.lmplot('iteration', 'ROC AUC', hue = 'search', data = scores, size = 8);
plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("ROC AUC versus Iteration");