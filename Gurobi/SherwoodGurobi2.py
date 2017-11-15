# -*- coding: utf-8 -*-
"""
"""

from gurobipy import *

# create Gurobi model
m = Model("sherwood")
m.ModelSense = GRB.MAXIMIZE
m.setParam('TimeLimit',7200)     #sets a time limit on execution to some number of seconds

dvars = []
# Set up decision variables
for i in range(2):
    dvars.append(m.addVar(vtype=GRB.CONTINUOUS,name='q'+str(i), lb=0.0))

#Update model to include variables
m.update()

cnstrt_coeff = [[4.0,3.0],[8.0,2.0],[0,1.0]]
cnstrt_names = ["AssemblyDeptHours","FinishingDeptHours","CustomSales"]
constr_rhs = [120.0,160.0,32.0]
obj_coeff = [20.0,10.0]

for i in range(len(cnstrt_coeff)):
    # Here's the template for adding a constraint
    # m.addConstr(4.0 * qs + 3.0 * qc, GRB.LESS_EQUAL, 120.0,"AssemblyDeptHours")
    m.addConstr(quicksum((cnstrt_coeff[i][j] * dvars[j] for j in range(len(dvars)))), GRB.LESS_EQUAL, constr_rhs[i],cnstrt_names[i])
# set up constraints
#m.addConstr(4.0 * qs + 3.0 * qc, GRB.LESS_EQUAL, 120.0,"AssemblyDeptHours")
#m.addConstr(8.0 * qs + 2.0 * qc, GRB.LESS_EQUAL, 160.0,"FinishingDeptHours")
#m.addConstr(qc, GRB.LESS_EQUAL, 32.0,"CustomSales")

# Set objective function
m.setObjective(quicksum(obj_coeff[i] * dvars[i] for i in range(len(dvars))), GRB.MAXIMIZE)

# Update the model
# It won't run if you don't do this
m.update()

# Optimize the model
m.optimize()

# alternate printout of results
for var in m.getVars():
    print "Variable Name = %s, Optimal Value = %s, Lower Bound = %s, Upper Bound = %s" % (var.varName, var.x,var.lb,var.ub)
    