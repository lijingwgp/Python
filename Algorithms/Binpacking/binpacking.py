# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:54:43 2018

@author: Jing
"""

def binpack(articles,bin_cap):
    """ You write your heuristic bin packing algorithm in this function using the argument values that are passed
             articles: a dictionary of the items to be loaded into the bins: the key is the article id and the value is the article volume
             bin_cap: the capacity of each (identical) bin (volume)
    
        Your algorithm must return two values as indicated in the return statement:
             my_team_number_or_name: if this is a team assignment then set this variable equal to an integer representing your team number;
                                     if this is an indivdual assignment then set this variable to a string with you name
             bin_contents: this is a list containing keys (integers) of the items you want to include in the knapsack
                           The integers refer to keys in the items dictionary. 
   """
        
    my_team_number_or_name = "jingli"    # always return this variable as the first item
    bin_contents = []    # use this list document the article ids for the contents of 
                         # each bin, the contents of each is to be listed in a sub-list
     
    new_articles = [[k,v] for k,v in articles.items()]
    a = lambda x:x[1]
    new_articles.sort(key=a)
    item_selected = []
     
    articles1 = new_articles[:(len(new_articles)-1)/2]
    articles2 = new_articles[(len(new_articles)-1)/2:]
    articles2.sort(key=a, reverse=True)

    while len(item_selected) != len(new_articles):
        bin_vol = []
        bin_key = []
        for this in articles2:
            if this[0] not in item_selected and (sum(bin_vol)+this[1]) <= bin_cap:
                    bin_vol.append(this[1])
                    bin_key.append(this[0])
                    item_selected.append(this[0])
            
        for that in articles1:
            if that[0] not in item_selected and (sum(bin_vol)+that[1]) <= bin_cap:
                bin_vol.append(that[1])
                bin_key.append(that[0])
                item_selected.append(that[0])
    
        bin_contents.append(bin_key)

    return my_team_number_or_name, bin_contents       # use this return statement when you have items to load in the knapsack
