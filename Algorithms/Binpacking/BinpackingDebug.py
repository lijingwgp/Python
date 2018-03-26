# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:31:22 2018

@author: Jing
"""

a = {0:1,2:32,3:20,4:12}
a1 = [[k,v] for k,v in a.items()]
a1

a = lambda x:x[1]
a1.sort(key=a)
a1

bin_cap = 34
bin_content = []
item_selected = []

# =============================================================================
# item_selected = []
# 
# while len(item_selected) != len(a1):
#     new_bin_vol = []
#     new_bin_key = []
#     for each in a1:
#         if each[0] not in item_selected and (sum(new_bin_vol)+each[1]) <= bin_cap:
#             new_bin_key.append(each[0])
#             new_bin_vol.append(each[1])
#             item_selected.append(each[0])
#     bin_content.append(new_bin_key)
# 
# =============================================================================

a11 = a1[:(len(a1)-1)/2]
a12 = a1[(len(a1)-1)/2:]
a12.sort(key=a, reverse=True)

while len(item_selected) != len(a1):
    bin_vol = []
    bin_key = []
    for this in a12:
        if this[0] not in item_selected and (sum(bin_vol)+this[1]) <= bin_cap:
                bin_vol.append(this[1])
                bin_key.append(this[0])
                item_selected.append(this[0])
                
    for that in a11:
        if that[0] not in item_selected and (sum(bin_vol)+that[1]) <= bin_cap:
            bin_vol.append(that[1])
            bin_key.append(that[0])
            item_selected.append(that[0])
    
    bin_content.append(bin_key)
                
    
    
    
    
    
    
    
    