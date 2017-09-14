#E360 v0.0.1
import sys
import numpy as np
import pandas as pd
import scipy.stats

def main():
	global data
	filename = sys.argv[1]
	data = pd.read_csv(filename)

	#Revenue Profit
	revenuePw = .25
	custPw = .15
	rev_employeePw = .15
	rev_employeeP_Ww = .15
	rev_custPw = .15
	volimpw = .15
	#Efficiency_Management
	fundPw = .15
	totalemployeePw = .15
	netemployeePw = .15
	meratioPw = .15
	meratioP_Ww = .15
	cust_employeePw = .25
	#Innovation
	innovationPw = .33
	aoiw = .33
	editorialw = .33
	#Pillars - Final Score
	rev_pillarw = .475
	eff_man_pillarw = .475
	inn_pillarw = .05

	#Define lookuptables
	dollar_di = {'0': 0,'Less than $50,000': .7,'$50,000 - $100,000': 1,'$100,001 - $500,000': 1.33,'$500,001 - $1,000,000': 1.67,'$1,000,001 - $2,500,000': 2,'$2,500,001 - $5,000,000': 2.33,'$5,000,001 - $7,500,000': 2.67,'$7,500,001 - $10,000,000': 3,'$10,000,001 - $25,000,000': 3.33,'$25,000,001 - $50,000,000': 3.67,'$50,000,001 - $100,000,000': 4,'More than $100,000,000': 4.33}
	custvol_di = {'0': 0,'Less than 49': .7,'50 - 99': 1,'100 - 249': 1.33,'250 - 499': 1.67,'500 - 999': 2,'1,000 - 4,999': 2.33,'5,000 - 9,999': 2.67,'10,000 - 24,999': 3,'25,000 - 49,999': 3.33,'50,000 - 99,999': 3.67,'100,000 - 199,999': 4,'200,000 - 499,999': 4.33,'500,000 - 749,999': 4.67,'750,000 - 999,999': 5,'1 million - 2.49 million': 5.33,'2.5 million - 4.9 million': 5.67,'5 million - 9.9 million': 6,'10 million - 19.9 million': 6.33,'20 million - 29.9 million': 6.67,'30 million - 50 million': 7,'More than 50 million': 7.33,}
	avgrev_di = {0:0,0.7:25000,1:75000,1.33:300000,1.67:750000,2:1750000,2.33:3750000,2.67:6250000,3:8750000,3.33:17500000,3.67:37500000,4:75000000,4.33:125000000}
	avgcust_di = {0:0,0.7:25,1:75,1.33:175,1.67:375,2:750,2.33:3000,2.67:7500,3:17500,3.33:37500,3.67:75000,4:150000,4.33:350000,4.67:625000,5:875000,5.33:1750000,5.67:3750000,6:7500000,6.33:15000000,6.67:25000000,7:40000000,7.33:75000000}

	#convert values using dictionaries
	data = data.replace({"historicalRevenue: 2015":dollar_di})
	data = data.replace({"historicalRevenue: 2014":dollar_di})
	data = data.replace({"historicalRevenue: 2013":dollar_di})
	data = data.replace({"funding":dollar_di})
	data = data.replace({"historicalCustomerCount: 2015":custvol_di})

	#New DataFrame Column based on dictionary
	data['avgrev'] = data['historicalRevenue: 2015'].map(avgrev_di)
	data['avgcust'] = data['historicalCustomerCount: 2015'].map(avgcust_di)
	#Convert null or NaN entries to 0
	data = data.replace(np.inf, 0)
	data = data.fillna(0)

	# Calculate New Datafields
	data['revenuescore'] = ((data['historicalRevenue: 2015']*5) + (data['historicalRevenue: 2014']*4) + (data['historicalRevenue: 2013']*3)) /12
	data['totalinnovation'] = data['historicalLegalCount: 2015'] + data['historicalLegalCount: 2014'] + data['historicalLegalCount: 2013']
	data['netemployee'] = data['historicalEmployeeCount: 2015'] - data['historicalEmployeeCount: 2013']
	data['meratio'] = data['historicalEmployeeCount: 2015'] / data['historicalManagementCount: 2015']
	data['rev_cust'] = data['avgrev'] / data['avgcust']
	data = data.replace(np.inf, 0)
	data = data.fillna(0)
	data['rev_employee'] = data['avgrev'] / data['historicalEmployeeCount: 2015']
	data['cust_employee'] = data['avgcust'] / data['historicalEmployeeCount: 2015']
	data = data.replace(np.inf, 0)
	data = data.fillna(0)

	#Calculate Pvalues
	Pvalue('custP','historicalCustomerCount: 2015')
	Pvalue('revenueP','revenuescore')
	Pvalue('fundP','funding')
	Pvalue('innovationP','totalinnovation')
	Pvalue('totalemployeeP','historicalEmployeeCount: 2015')
	Pvalue('netemployeeP','netemployee')
	Pvalue('meratioP','meratio')
	Pvalue('rev_custP','rev_cust')
	Pvalue('rev_employeeP','rev_employee')
	Pvalue('cust_employeeP','cust_employee')
	Pvalue('eisP','editInnovationScore')
	Pvalue('operatingCostP','operatingCost')

	PvalueW('meratioP_W','meratio')
	PvalueW('rev_employeeP_W','rev_employee')

	data['aoi_adj'] = data.apply(Adjust_aoi, axis=1)
	data['aoiScore'] = np.multiply(data['aoi_adj'],data.groupby('industry').std().loc[data.loc[:,'industry'],'revenueP'].replace(0,0.000001).replace(np.nan,0.000001))

	#Pillar Calculations
	data['Rev_Profit_Pillar'] = (data['revenueP']*revenuePw)+(data['custP']*custPw)+(data['rev_employeeP']*rev_employeePw)+(data['rev_employeeP_W']*rev_employeeP_Ww)+(data['rev_custP']*rev_custPw)+((1-data['operatingCostP'])*volimpw) #(data['operatingCost']/100)
	data['Eff_Management'] = (data['fundP']*fundPw)+(data['totalemployeeP']*totalemployeePw)+(data['netemployeeP']*netemployeePw)+(data['meratioP']*meratioPw)+(data['meratioP_W']*meratioP_Ww)+(data['cust_employeeP']*cust_employeePw)
	data['InnovationScore'] = (data['innovationP']*innovationPw)+(data['aoiScore']*aoiw)+(data['eisP']*editorialw)
	data['FinalScore'] = (data['Rev_Profit_Pillar']*rev_pillarw) + (data['Eff_Management']*eff_man_pillarw) + (data['InnovationScore']*inn_pillarw)

	data.to_csv('E360-Ranking.csv', encoding='utf-8')

def Pvalue(new_field, source_field):
    data[new_field] = scipy.stats.norm(data.groupby('industry').mean().loc[data.loc[:,'industry'],source_field], data.groupby('industry').std().loc[data.loc[:,'industry'],source_field].replace(0,0.000001).replace(np.nan,0.000001)).cdf(data[source_field])

def PvalueW(new_field, source_field):
	data[new_field] = scipy.stats.norm(data[source_field].mean(), data[source_field].std()).cdf(data[source_field])

def Adjust_aoi(row):
	if row['ageIndustry'] == 1:
		if row['revenueP'] > .9:
			return .2
		elif row['revenueP'] > .1 and row['revenueP'] < .9:
			return .1
		else:
			return 0
	elif row['ageIndustry'] == 2:
		if row['revenueP'] > .95:
			return .1
		else:
			return 0
	elif row['ageIndustry'] == 3:
		return 0
	elif row['ageIndustry'] == 4:
		if row['revenueP'] > .95:
			return 0
		elif row['revenueP'] > .05 and row['revenueP'] < .95:
			return -.1
		else:
			return -.2
	else:
		if row['revenueP'] > .9:
			return 0
		else:
			return -.2

if __name__ == '__main__': main()