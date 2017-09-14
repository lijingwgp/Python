# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 21:40:34 2017

@author: jrbrad
"""

import requests
from bs4 import BeautifulSoup as bs

html_doc = requests.get("http://www.twitter.com","lxml").content
parsed_html = bs(html_doc)
targets = parsed_html.find_all('div', attrs={'class' : 'MomentCapsuleSummary MomentCapsuleSummary--card'})

for row in targets:
    #print type(row)
    print row['data-moment-id']
    #title = row.find_all('a', attrs={'class' : "MomentCapsuleSummary-title u-textUserColorHover js-default-link js-nav u-dir"})
    title = row.find('a', attrs={'class' : "MomentCapsuleSummary-title u-textUserColorHover js-default-link js-nav js-ellipsis u-dir"})
    
    print title.text.strip()
    category = row.find('span', attrs = {'class':"MomentCapsuleSubtitle-category u-dir"})
    print category.text.strip()
    print
#print len(targets), type(targets[0])