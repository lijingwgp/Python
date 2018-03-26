# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 09:21:52 2016

@author: james.bradley
"""

from gurobipy import *
import MySQLdb as mySQL 

mysql_user_name =  'root'
mysql_password = 'root'
mysql_ip = '127.0.0.1'
mysql_db = 'knapsack'

def db_get_data(problem_id):
    cnx = mySQL.connect(user='root', passwd='root',
                        host='127.0.0.1', db='knapsack')
                        
    cursor = cnx.cursor()
    cursor.execute("CALL spGetKnapsackCap(%s);" % problem_id)
    knap_cap = cursor.fetchall()[0][0]
    cursor.close()
    cursor = cnx.cursor()
    cursor.execute("CALL spGetKnapsackData(%s);" % problem_id)
    items = {}
    blank = cursor.fetchall()
    for row in blank:
        items[row[0]] = (row[1],row[2])
    cursor.close()
    cnx.close()
    return knap_cap, items

def getDBDataList(commandString):
    cnx = db_connect()
    cursor = cnx.cursor()
    cursor.execute(commandString)
    items = []
    for item in list(cursor):
        items.append(item[0])
    cursor.close()
    cnx.close()
    return items

def db_connect():
    cnx = mySQL.connect(user=mysql_user_name, passwd=mysql_password,
                        host=mysql_ip, db=mysql_db)
    return cnx
    
#problem_id = 0
problems = getDBDataList('CALL spGetProblemIds();')

# create Gurobi model
m = Model("Knapsack")
m.ModelSense = GRB.MAXIMIZE
    
for problem_id in problems:
    knap_cap, items = db_get_data(problem_id)    

    print
    print
    print "=================================="
    print
    print "Problem",problem_id
    
    """ these reset functions are not required
    m.remove(model.getVars())
    m.remove(model.getConstrs())
    m.remove(model.getObjective())
    m.update()
    """
    
    x = {}
    for thisKey in items.keys():
        x[thisKey] = m.addVar(vtype=GRB.BINARY,name='x'+str(thisKey))
    
    m.update()
    
    m.addConstr(quicksum(x[thisKey] * items[thisKey][0] for thisKey in x.keys()), GRB.LESS_EQUAL, knap_cap)
        
    m.setObjective(quicksum(x[thisKey] * items[thisKey][1] for thisKey in x.keys()), GRB.MAXIMIZE)
    
    m.update()
    m.optimize()
    
    for thisKey in x.keys():
        if x[thisKey].x > 0:
            print "Item ", str(thisKey), ": ",items[thisKey]
                
       
    print 'Optimal Objective Function Value for Problem ',problem_id,':',m.ObjVal