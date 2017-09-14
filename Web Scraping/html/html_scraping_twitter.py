# -*- coding: utf-8 -*-
"""
Created on Thu May 26 21:29:00 2016

@author: james.bradley
"""

"""
Based on this web site:
    http://stackoverflow.com/questions/11709079/parsing-html-using-python
"""

#import re
import requests
from bs4 import BeautifulSoup

html_path = 'https://twitter.com'
html_doc = requests.get(html_path, "lxml").content

# parse html 
parsed_html = BeautifulSoup(html_doc)

target_rows = parsed_html.find_all('div', attrs={'class' : 'MomentCapsuleSummary MomentCapsuleSummary--card'})
#print stats
print 'Number of Moments found:',len(target_rows)
print 'Stats are in data type:',type(target_rows)
print
all_mods = []
for row in target_rows:
    new_row = []
    new_row.append(row['data-moment-id'])
    for x in row.find_all('div', attrs={'class' : 'MomentCapsuleSubtitle'}):
        for y in x.find_all('span' , attrs={'class' : 'MomentCapsuleSubtitle-category u-dir'}):
            new_row.append(y.text.encode("ascii",'ignore'))    #x.text.encode("ascii",'ignore')
    for z in row.find_all('div', attrs = {'class' : 'MomentCapsuleSummary-details'}):
        new_row.append(z.text.encode("ascii",'ignore'))
        
    all_mods.append(new_row)
    
print '\nHere\'s the Data'
print all_mods