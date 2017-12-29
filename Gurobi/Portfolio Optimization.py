# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 09:34:06 2017

@author: Jing
"""
from gurobipy import *
import MySQLdb as mySQL
db = mySQL.connect(user='root', passwd='root', host='localhost', db='nasdaq')



cur = db.cursor()
#cur.execute('select * from corr')
#print cur.description
#ncol = len(cur.description)
#cols = [cur.description[i][0] for i in range(0, ncol)]



cur.execute('select * from return_rate')
ret = cur.fetchall()
avg_ret = []
for i in range(1158):
    avg_ret.append(ret[i][0])



cur.execute('select cov from covariance')
cov = cur.fetchall()
#q = np.zeros((1158,1158),dtype=float)
#q = np.zeros(1158*1158,dtype=float)
#q = {}
#k=0
#for i in range(5):
#    for j in range(5):
#        q[(i,j)] = cov[k][0]
#        k=k+1
Q = []
k=0
for i in range(1158):
    Q_new = []
    for j in range(1158):
        Q_new.append(cov[k][0])
        k=k+1
    Q.append(Q_new)
    


m = Model("Optimal Portfolio_Non-Linear")
m.ModelSense = GRB.MAXIMIZE
m.setParam('TimeLimit',7200)



a = []
for i in range(1158):
    a.append(m.addVar(vtype=GRB.CONTINUOUS,name='a'+str(i), lb=0.0))
m.update



port_size = 1
max_risk = 0.00102
m.addConstr(quicksum(a[i]*Q[i][j]*a[j] for i in range(1158) for j in range(1158)), GRB.LESS_EQUAL, max_risk,'ControlRisk')
m.addConstr(quicksum(a[i]*1 for i in range(1158)), GRB.EQUAL, port_size, 'PortfolioSum')
m.update



combination = []
incr = 0.01788
for i in range(25):
    m.setObjective(quicksum(avg_ret[i]*a[i] / port_size for i in range(1158)), GRB.MAXIMIZE)
    m.update()
    m.optimize()
    
    combination_temp = []
    combination_temp.append(max_risk)
    combination_temp.append(m.ObjVal)
    combination.append(combination_temp)
    
    #m.getQConstrs()
    m.remove(m.getQConstrs())
    m.update()
    
    max_risk = max_risk + incr
    m.addConstr(quicksum(a[i]*Q[i][j]*a[j] for i in range(1158) for j in range(1158)), GRB.LESS_EQUAL, max_risk,'ControlRisk')
    m.update()



cur.execute('create table combination(risk double, expected_return double)')
for i in range(25):
    cur.execute('insert into combination values (%s,%s)',(combination[i][0],combination[i][1]))
    db.commit()
