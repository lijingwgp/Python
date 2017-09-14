# F500 Sublist 1 -- Fastest Growing Franchises in North America
# Method: Rank based on US and Canada raw unit growth, break tie with US and Canada percentage growth

import sys
import numpy as np
import pandas as pd
import scipy.stats

# Import data from franchise500 ranking
def main():
	global data
	filename = sys.argv[1]
	data = pd.read_csv(filename)
	
	selected_data = data.loc[data['US Units 2016'] >= 1]

	# Calculating raw unit growth for US and Canada
	data['US Added'] = selected_data['US Units 2016'] - selected_data['US Units 2015']
	data['CAN Added'] = selected_data['CAN Units 2016'] - selected_data['CAN Units 2015']
	data['Total Added'] = data['US Added'] + data['CAN Added']

	# Calculating percentage growth for US and Canada
	data['Percent Growth'] = data['Total Added'] / (selected_data['US Units 2015'] + selected_data['CAN Units 2015'])
	data = data.replace(np.inf, 0)
	data = data.fillna(0)

	data.to_csv('Fastest-Growing-Franchise.csv', encoding='utf-8')
    
if __name__ == '__main__': main()