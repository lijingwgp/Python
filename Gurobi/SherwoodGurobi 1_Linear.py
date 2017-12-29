# -*- coding: utf-8 -*-

from gurobipy import *

# create Gurobi model
m = Model("sherwood")
m.ModelSense = GRB.MAXIMIZE
m.setParam('TimeLimit',7200)     #sets a time limit on execution to some number of seconds

# Set up decision variables
qs = m.addVar(vtype=GRB.CONTINUOUS,name='qs', lb=0.0)
qc = m.addVar(vtype=GRB.CONTINUOUS,name='qc', lb=0.0, ub=32.0)

#Update model to include variables
m.update()

# set up constraints
m.addConstr(4.0 * qs + 3.0 * qc, GRB.LESS_EQUAL, 120.0,"AssemblyDeptHours")
m.addConstr(8.0 * qs + 2.0 * qc, GRB.LESS_EQUAL, 160.0,"FinishingDeptHours")
m.addConstr(qc, GRB.LESS_EQUAL, 32.0,"CustomSales")

# Set objective function
m.setObjective(20.0 * qs + 10.0 * qc, GRB.MAXIMIZE)

# Update the model
# It won't run if you don't do this
m.update()

# Optimize the model
m.optimize()

# Print solution: .x gives the optimal objective function value
print (qs.varName, "=", qs.x, "(", qs.lb,",",qs.ub,")")
print (qc.varName, "=", qc.x, "(", qc.lb,",",qc.ub,")")

# alternate printout of results
for var in m.getVars():
    print ("Variable Name = %s, Optimal Value = %s, Lower Bound = %s, Upper Bound = %s" % (var.varName, var.x,var.lb,var.ub))
    
