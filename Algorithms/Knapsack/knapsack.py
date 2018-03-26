# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:28:13 2018

@author: Jing
"""

def load_knapsack(things,knapsack_cap):
    """ You write your heuristic knapsack algorithm in this function using the argument values that are passed
             items: is a dictionary of the items not yet loaded into the knapsack
             knapsack_cap: the capacity of the knapsack (volume)
    
        Your algorithm must return two values as indicated in the return statement:
             my_team_number_or_name: if this is a team assignment then set this variable equal to an integer representing your team number;
                                     if this is an indivdual assignment then set this variable to a string with you name
             items_to_pack: this is a list containing keys (integers) of the items you want to include in the knapsack
                            the integers refer to keys in the items dictionary. 
   """
        
    my_team_number_or_name = "jingli"    # always return this variable as the first item
    items_to_pack = []    # use this list for the indices of the items you load into the knapsack
    load = 0.0            # use this variable to keep track of how much volume is already loaded into the backpack
    value = 0.0           # value in knapsack
    
    new_things = [[k,v] for k,v in things.items()]
    new_things1 = list()
    for each in new_things:
        new_things1.append([each[0],each[1],each[1][1] / each[1][0]])
    
    a = lambda x:x[2]
    new_things1.sort(key=a, reverse=True)
    
    min_volume = min(each[1][0] for each in new_things1)
    residual_volume = 0

    for i in range(len(new_things1)):
        residual_volume = knapsack_cap - load
        if residual_volume < min_volume:
            return my_team_number_or_name, items_to_pack
        elif residual_volume < new_things1[i][1][0]:
            continue
        else:
            item_selected = new_things1[i][0]
            items_to_pack.append(item_selected)
            load += new_things1[i][1][0]
            value += new_things1[i][1][1]

    return my_team_number_or_name, items_to_pack    # use this return statement when you have items to load in the knapsack
    