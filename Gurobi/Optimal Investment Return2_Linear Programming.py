# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:55:38 2017

@author: Jing
"""

from gurobipy import *

# create Gurobi model
m = Model("Optimal Investment Return 2_Linear")
m.ModelSense = GRB.MINIMIZE
m.setParam('TimeLimit',7200)     #sets a time limit on execution to some number of seconds

# Set up decision variables
dvars = []
for i in range(8):
    dvars.append(m.addVar(vtype=GRB.INTEGER,name='x'+str(i), lb=0.0))

# Update model to include variables
m.update()

# Given data
risk = [0.0, 0.0, 25.0, 30.0, 20.0, 15.0, 65.0, 40.0]
money_return = [.04, .052, .071, .1, .082, .065, .2, .125]

# Set up constrains
for i in range(len(dvars)):
    m.addConstr(quicksum(dvars[i]*1 for i in range(len(dvars))), GRB.EQUAL, 100000.0,'TotalFundsAvaliable')
    m.addConstr(quicksum(money_return[i]*dvars[i] for i in range(len(money_return))), GRB.EQUAL, 7500.0,'ExpectedReturn')

m.addConstr(dvars[0] + dvars[1] + dvars[4] + dvars[6], GRB.GREATER_EQUAL, 50000.0, "AtLeast50A-Rated")
m.addConstr(dvars[0] + dvars[2] + dvars[3] + dvars[6] + dvars[7], GRB.GREATER_EQUAL, 40000.0, 'AtLeast40Immediate')
m.addConstr(dvars[0] + dvars[1], GRB.LESS_EQUAL, 30000.0, 'NoMoreThan30000')

# Set objective function
m.setObjective(quicksum(risk[i] * dvars[i] for i in range(len(dvars))), GRB.MINIMIZE)

# Update the model
m.update()

# Optimize the model
m.optimize()

# printout of results
for var in m.getVars():
    print ("Variable Name = %s, Optimal Value = %s" % (var.varName, var.x))