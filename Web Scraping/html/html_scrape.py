# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:02:01 2017

@author: jrbrad
"""

import requests
from bs4 import BeautifulSoup as bsoup
    
my_wm_username = 'jli23'
search_url = 'http://publicinterestlegal.org/county-list/'
response = requests.get(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"}).content
            
# Put your program here
parsed_html = bsoup(response, 'lxml')
target_rows = parsed_html.find_all('tr')

big = []
for row in target_rows:
    small = []
    for x in row.find_all('td'):
        small.append(x.text.encode("ascii",'ignore'))
    big.append(small)

print my_wm_username
print len(big)
print big