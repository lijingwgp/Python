# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:08:23 2018

@author: Jing
"""
# after the conversion of dictionary to list
test = [[0,(3,9)],[1,(1,5)]]
new_things1 = list()
for each in test:
    new_things1.append([each[0],each[1],each[1][1] / each[1][0]])
new_things1

a = lambda x:x[2]
new_things1.sort(key=a, reverse=True)
new_things1

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




things={0:(1,2), 1:(2,8)}
myDict = {}
for Object in things.keys():
    myDict[Object] = (things[Object][0],things[Object][1],things[Object][1]/things[Object][0])
    Items = [[a,b] for a,b in myDict.items()]
Items

Ratio = lambda x:x[1][2]
Items.sort(key = Ratio, reverse = True)
Items

MinVolume = min(i[1][0] for i in Items)
Leftovers = 0
for i in range(len(Items)):
    Leftovers = knapsack_cap - load
    if Leftovers < MinVolume:
        return my_team_number_or_name, items_to_pack
    elif Leftovers < Items[i][1][0]:
        continue
    else:
        pack_item = Items[i][0]
        items_to_pack.append(pack_item)
        load += Items[i][1][0]
        value += Items[i][1][1]
     