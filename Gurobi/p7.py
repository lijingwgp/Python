# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:12:22 2017

@author: Jing
"""

from gurobipy import *
import MySQLdb as mySQL
m = Model("Optimize Transportation Cost")
m.ModelSense = GRB.MINIMIZE
m.setParam('TimeLimit',7200)


db = mySQL.connect(user='root', passwd='root', host='localhost', db='transportation')
cur = db.cursor()
cur.execute('select mileage from mileage')



supply_decision = []
for i in range(10):
    temp = []
    for j in range(1100):
        temp.append(m.addVar(vtype=GRB.BINARY,name='dc'+str(i)+'_st'+str(j)))
    supply_decision.append(temp)



supply_amount = []
for i in range(10):
    temp = []
    for j in range(1100):
        temp.append(m.addVar(vtype=GRB.INTEGER,name='dc'+str(i)+'_st'+str(j)+'_amount', lb=0.0))
    supply_amount.append(temp)
m.update()



mileage = []
for i in range(10):
    temp = []
    for j in range(1100):
        mil = cur.fetchone()
        temp.append(mil[0])
    mileage.append(temp)



store_demand = []
cur.execute('select requirements from store')
st = cur.fetchall()
for i in range(1100):
    store_demand.append(st[i][0])



individual_demand = []
for i in range(10):
    temp = []
    k = 0
    for j in range(1100):
        temp.append(supply_decision[i][j]*store_demand[k])
        k = k+1
    individual_demand.append(temp)
for i in range(10):
    for j in range(1100):
        m.addConstr(supply_amount[i][j],GRB.GREATER_EQUAL,individual_demand[i][j],'IndividualDemand')
m.update()



for i in range(10):
    m.addConstr(quicksum(supply_amount[i][j] for j in range(1100)),GRB.LESS_EQUAL,12000,'CapacityConstrain')
m.update()



unique = []
for i in range(1100):
    temp = []
    for j in range(10):
        temp.append(supply_decision[j][i])
    unique.append(temp)
for i in range(1100):
    m.addConstr(quicksum(unique[i][j]*1 for j in range(10)),GRB.EQUAL,1,'Uniqueness')
m.update()



total_demand = []
for i in range(10):
    m.addConstr(quicksum(individual_demand[i][j] for j in range(1100)),GRB.LESS_EQUAL,quicksum(individual_demand[i][j] for j in range(1100)),'DemandConstrain')
m.update()



m.setObjective(quicksum(supply_amount[i][j]*mileage[i][j] for i in range(10) for j in range(1100))*0.75+quicksum(supply_amount[i][j]*200 for i in range(10) for j in range(1100)),GRB.MINIMIZE)
m.update()
m.optimize()



value = []
for i in range(11000):
    if m.getVars()[i].x > 0.0:
        value.append(m.getVars()[i].varname)



dc = []
for i in range(1100):
    dc.append(value[i][2])
store = []
for i in range(1100):
    store.append(value[i][6:])



for i in range(1100):
    cur.execute('insert into results values (%s,%s)',(dc[i],store[i]))
    db.commit()
