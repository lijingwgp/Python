# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:32:02 2016
@author: james.bradley
"""


import requests
from lxml import objectify

periods = '6'
state = '44'
div = '0'
month = '8'
year = '2016'

url_base = 'https://www.ncdc.noaa.gov/temp-and-precip/climatological-rankings/download.xml?'
url_add = 'parameter=tavg&state=%s&div=%s&month=%s&periods[]=%s&year=%s'

temp_data = (state,div,month,periods,year)
url = url_base + url_add % temp_data

response = requests.get(url).content
root = objectify.fromstring(response)

my_wm_username = 'jli23'

print my_wm_username
print root['data']['value']
print root['data']['twentiethCenturyMean']
print root['data']['lowRank']
print root['data']['highRank']