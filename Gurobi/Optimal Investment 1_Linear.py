# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:13:24 2017

@author: Jing
"""

from gurobipy import *

# create Gurobi model
m = Model("Optimal Investment Return 1_Linear")
m.ModelSense = GRB.MINIMIZE
m.setParam('TimeLimit',7200)     #sets a time limit on execution to some number of seconds

# Set up decision variables
saving = m.addVar(vtype=GRB.INTEGER, name='saving', lb=0.0)
cd = m.addVar(vtype=GRB.INTEGER, name='cd', lb=0.0)
atlantic = m.addVar(vtype=GRB.INTEGER, name='atlantic', lb=0.0)
arkansas = m.addVar(vtype=GRB.INTEGER, name='arkansas', lb=0.0)
bedrock = m.addVar(vtype=GRB.INTEGER, name='bedrock', lb=0.0)
nocal = m.addVar(vtype=GRB.INTEGER, name='nocal', lb=0.0)
minicomp = m.addVar(vtype=GRB.INTEGER, name='minicomp', lb=0.0)
antony = m.addVar(vtype=GRB.INTEGER, name='antony', lb=0.0)

# Update model to include variables
m.update()

# set up constraints
m.addConstr(saving + cd + atlantic + arkansas + bedrock + nocal + minicomp + antony, 
            GRB.EQUAL, 100000.0, "TotalAvaliableFunds")
m.addConstr(.04*saving + .052*cd + .071*atlantic + .1*arkansas + .082*bedrock +
            .065*nocal+.2*minicomp+.125*antony, GRB.GREATER_EQUAL, 7500.0, "ExpectedAnnualReturn")
m.addConstr(saving + cd + bedrock + minicomp, GRB.GREATER_EQUAL, 50000.0, "AtLeast50A-Rated")
m.addConstr(saving + atlantic + arkansas + minicomp + antony, GRB.GREATER_EQUAL, 40000.0, 'AtLeast40Immediate')
m.addConstr(saving + cd, GRB.LESS_EQUAL, 30000.0, 'NoMoreThan30000')

# Set objective function
m.setObjective(0*saving + 0*cd + 25*atlantic + 30*arkansas + 20*bedrock + 
               15*nocal + 65*minicomp + 40*antony, GRB.MINIMIZE)

# Update the model
m.update()

# Optimize the model
m.optimize()

# printout of results
for var in m.getVars():
    print ("Variable Name = %s, Optimal Value = %s" % (var.varName, var.x))
