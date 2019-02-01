# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 13:12:43 2019

@author: jing.o.li
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
x = data[feature_names]
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_x, train_y)

# visualize a single decision tree
from sklearn import tree
import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)

# note that the pair of values at the bottom show the count of true values and false values
# for the target respectively, of data point in that node of the tree.

# now we are going to use the PDPBox library to create partial dependence plots

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_x, model_features=feature_names,
                            feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show() 

# a few items are worth pointing out as we interpret this plot
# y axis is interpreted as change in the prediction 
# a blue shaded area indicates level of confidence

# from this particular graph, we see that scoring a goal substantially increases your chances
# of winning "MVP" award. But extra goals beyond that appear to have little impact on predictions

pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=val_x, model_features=feature_names,
                           feature='Distance Covered (Kms)')
pdp.pdp_plot(pdp_dist, 'Distance Covered (Kms)')
plt.show()

# we can also use more advanced models like random forest
rf = RandomForestClassifier().fit(train_x, train_y)
pdp_dist = pdp.pdp_isolate(model=rf, dataset=val_x, model_features=feature_names,
                           feature='Distance Covered (Kms)')
pdp.pdp_plot(pdp_dist, 'Distance Covered (Kms)')
plt.show()

# this model thinks you are more likely to win the award if your player run a total of 
# 100km over the course of the game. though running much more causes lower predictions

# if we are curious about interactions between features, 2D partial dependence plots are also
# useful. An example may clarify what this is.

features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_x, model_features=feature_names, features=features_to_plot)
pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()
