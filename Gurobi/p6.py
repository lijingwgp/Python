# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 02:10:03 2017

@author: Jing
"""

from gurobipy import *
import MySQLdb as mySQL
m = Model("Optimize Transportation Cost")
m.ModelSense = GRB.MINIMIZE
m.setParam('TimeLimit',7200)


db = mySQL.connect(user='root', passwd='root', host='localhost', db='transportation')
cur = db.cursor()
cur.execute('select mileage from mileage where dc_id <4 and store_id < 4')
mil = cur.fetchall()



supply_decision = []
for i in range(4):
    temp = []
    for j in range(4):
        temp.append(m.addVar(vtype=GRB.BINARY,name='dc'+str(i)+'_st'+str(j)+'_decision'))
    supply_decision.append(temp)



supply_amount = []
for i in range(4):
    temp = []
    for j in range(4):
        temp.append(m.addVar(vtype=GRB.INTEGER,name='dc'+str(i)+'_st'+str(j)+'_amount', lb=0.0))
    supply_amount.append(temp)
m.update()



k=0
mileage = []
for i in range(4):
    temp = []
    for j in range(4):
        temp.append(mil[k][0])
        k=k+1
    mileage.append(temp)



store_demand = [208,54,66,282]
individual_demand = []
for i in range(4):
    temp = []
    k = 0
    for j in range(4):
        temp.append(supply_decision[i][j]*store_demand[k])
        k = k+1
    individual_demand.append(temp)
for i in range(4):
    for j in range(4):
        m.addConstr(supply_amount[i][j],GRB.GREATER_EQUAL,individual_demand[i][j],'IndividualDemand')
m.update()



for i in range(4):
    m.addConstr(quicksum(supply_amount[i][j] for j in range(4)),GRB.LESS_EQUAL,12000,'CapacityConstrain')
m.update()



unique = []
for i in range(4):
    temp = []
    for j in range(4):
        temp.append(supply_decision[j][i])
    unique.append(temp)
for i in range(4):
    m.addConstr(quicksum(unique[i][j]*1 for j in range(4)),GRB.EQUAL,1,'Uniqueness')
m.update()



total_demand = []
for i in range(4):
    m.addConstr(quicksum(individual_demand[i][j] for j in range(4)),GRB.LESS_EQUAL,quicksum(individual_demand[i][j] for j in range(4)),'DemandConstrain')
m.update()



m.setObjective(quicksum(supply_amount[i][j]*mileage[i][j] for i in range(4) for j in range(4))*0.75+quicksum(supply_amount[i][j]*200 for i in range(4) for j in range(4)),GRB.MINIMIZE)
m.update()
m.optimize()
