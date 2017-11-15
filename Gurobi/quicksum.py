# -*- coding: utf-8 -*-
"""
"""

from gurobipy import *

constr_rhs = [120.0,160.0,32.0]
mult = [2.0,2.0,2.0]
x = quicksum(constr_rhs[i] * mult[i] for i in range(len(constr_rhs)))
print (x.getValue())