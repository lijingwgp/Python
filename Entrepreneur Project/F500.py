#!/usr/bin/env python
#F500 v0.0.5
import sys
import numpy as np
import pandas as pd
import scipy.stats
import tweepy
import facebook

def main():
	global data
	filename = sys.argv[1]
	data = pd.read_csv(filename)

	#General Settings
	currentYear = 2016
	global YrsBtwFranchise_threshold
	global YrsFranchising_threshold
	global YrsinBusiness_threshold
	global FeeRatio_Penalty_weight
	global FeeRatio_Penalty_percent
	global SlowGrowthPenaltyUnits
	global SlowGrowth_Penalty_weight 
	YrsBtwFranchise_threshold = 3
	YrsFranchising_threshold = 3
	YrsinBusiness_threshold = 3
	Penaltyweight = 2  # multiplied by STDEV

	#Weightings
	FranchiseTraining_w = 1.2
	LeaseNegotiation_w = 4

	#P1 (.16)
	FranchiseFee_w = .2
	FranchiseRoyalty_w = 0
	AdRoyalty_w = 0
	AverageTotalInvestment_w = .5
	TotalRoyalty_w = .3
	ResidualCapitalAssets_w = .1 #Not Used
	
	FeeRatio_Penalty_percent = .4
	FeeRatio_Penalty_weight = 1

	#P2 (.1)
	USUnits_w = .20
	CanadaUnits_w = .05
	ForeignUnits_w = .025
	CompanyUnits_w = .025
	Terminations_w = .10
	Reacquire_w = .00
	NonRenewal_w = .00
	Ceased_w = .05
	AverageGrowthRate_w = .15
	AdjustedUnits_w = .40

	#P3 (.14)
	TotalTrainingDays_w = .25
	FranchiseEmployee_w = .10
	Litigations_w = .20
	InHouseFinance_w = .15
	ThirdPartyFinance_w = .20
	OngoingSupport_w = .05
	MarketingSupport_w = .05

	#P4 (.20)
	UnitsPerYear_w = .20
	SocialScore_w = .20
	Totalunits_w = .20
	YrsinBusiness_w = .20
	YrsFranchising_w = .20

	
	SlowGrowthPenaltyUnits = 3
	SlowGrowth_Penalty_weight = 1

	#P5
	CurrentRatio_w = .40
	DebtEquityRatio_w = .15
	MemberEquityRatio_w = .15
	ProfitMargin_w = .30

	#Pillar Weightings (.2)
	p1_w = .10
	p2_w = .30
	p3_w = .15
	p4_w = .25
	p5_w = .20

	YNXrep(data.columns[38:64].values)   # start on the accurate column, end one after.
	data = data.replace(np.inf, 0)
	data = data.fillna(0)

	#Pillar 1
	data['Average FranchiseFee'] = (data['FranchiseFeeLow'] + data['FranchiseFeeHigh'])/2
	data['Average FranchiseTot'] = (data['FranchiseTotLow'] + data['FranchiseTotHigh'])/2
	data['TotalRoyalty'] = data['FranchiseRoyalty_score'] + data['FranchiseAdRoyalty_score']
	CPV('FranchiseFeeP','Average FranchiseFee','rankingCategory')
	PV('FranchiseRoyaltyP','FranchiseRoyalty_score')
	PV('AdRoyaltyP','FranchiseAdRoyalty_score')
	CPV('Average FranchiseTotP', 'Average FranchiseTot','rankingCategory')
	CPV('TotalRoyaltyP', 'TotalRoyalty','rankingCategory')
	data['FranchiseFee_Ratio'] = data['Average FranchiseFee'] / data['Average FranchiseTot']
	data['FranchiseFee_Ratio_Penalty'] = data.apply(FranchiseFeePenalty, axis=1)
	data['FranchiseFeeP_Final'] = (1-data['FranchiseFeeP']) - data['FranchiseFee_Ratio_Penalty']

	#Pillar 2
	data['TotalUnits2016'] = (data['US Units 2016'] + data['CAN Units 2016'] + data['FGN Units 2016'] + data['COM Units 2016'])
	data['TotalUnits2015'] = (data['US Units 2015'] + data['CAN Units 2015'] + data['FGN Units 2015'] + data['COM Units 2015'])
	data['yoyUSGrowth'] = (data['US Units 2016'] - data['US Units 2015'])
	data['3yrUSGrowth'] = (data['US Units 2016'] - data['US Units 2014'])
	data['TerminationsPercent'] = (data['Terminations'] / data['TotalUnits2015'])
	data['ReacquirePercent'] = (data['ReAcquired'] / data['TotalUnits2015'])
	data['NonRenewPercent'] = (data['NonRenewal'] / data['TotalUnits2015'])
	data['CeasedPercent'] = (data['CeasedOperations'] / data['TotalUnits2015'])

	data['USOldestYear'] = data.apply(Year_calc, args=(data.columns[94:104].values,), axis=1)
	data['USNumberofYears'] = (data.loc[:,data.columns[94:104]] != 0).astype(int).sum(axis=1)
	data['AverageGrowthRate'] = ((data['US Units 2016'] - data['USOldestYear'])/data['USOldestYear'])/ data['USNumberofYears']
	data = data.fillna(0)
	data['AverageGrowthRateFinal'] = data.apply(FiveYearGrowthBonus, axis=1)

	PV('USUnits2016P', 'US Units 2016')
	PV('CanUts15P', 'CAN Units 2016')
	PV('ForUts15P', 'FGN Units 2016')
	PV('ComUnits15P', 'COM Units 2016')
	PV('TerminationsP', 'TerminationsPercent')
	PV('ReacquireP', 'ReacquirePercent')
	PV('NonRenewP', 'NonRenewPercent')
	PV('CeasedP', 'CeasedPercent')
	PV('AverageGrowthRateP', 'AverageGrowthRateFinal')

	data['InvestmentVsCategoryAvg'] = np.divide(data['Average FranchiseTot'],data.groupby('rankingCategory').std().loc[data.loc[:,'rankingCategory'],'Average FranchiseTot'].replace(0,0.000001).replace(np.nan,0.000001))
	data['USGrowthVsCategoryAvg'] = np.divide(data['3yrUSGrowth'],data.groupby('rankingCategory').std().loc[data.loc[:,'rankingCategory'],'3yrUSGrowth'].replace(0,0.000001).replace(np.nan,0.000001))
	data['AdjustedGrowthRank'] = data.apply(AdjustedUnitGrowthClassification, axis=1)
	data['AdjustedUnits'] = data.apply(AdjustedUnitGrowthCalc, axis=1)
	PV('AdjustedUnitsP', 'AdjustedUnits')
	data['AdjustedUnitsFinal'] = data.apply(FinScore_adj, args=('AdjustedUnitsP',), axis=1)

	#Pillar 3
	data['TotalTrainingDays'] = data['TrngAtHQNoof_score'] + (data['TrngAtFraNoofDays_score'] * FranchiseTraining_w)
	data['LeaseNegotiationYN'] = data['LeaseNegotiationYN'] * LeaseNegotiation_w
	data['FranchiseEmployeePerUnit'] = data['TotalUnits2016'] / data['NoEmpFran']
	data['LitigationsPerUnit'] = data['Litigation'] / data['TotalUnits2016']
	data = data.replace(np.inf, 0)
	data['InHouseFinanceScore'] = (data.loc[:,data.columns[38:44]] != 0).astype(int).sum(axis=1)
	data['ThirdPartyFinanceScore'] = (data.loc[:,data.columns[45:51]] != 0).astype(int).sum(axis=1)
	data['OngoingSupportScore'] = (data.loc[:,data.columns[51:60]] != 0).astype(int).sum(axis=1)
	data['MarketingScore'] = (data.loc[:,data.columns[60:64]] != 0).astype(int).sum(axis=1)

	PV('TotalTrainingDaysP', 'TotalTrainingDays')
	PV('FranchiseEmployeePerUnitP', 'FranchiseEmployeePerUnit')
	PV('LitigationsPerUnitP', 'LitigationsPerUnit')
	PV('InHouseFinanceScoreP', 'InHouseFinanceScore')
	PV('ThirdPartyFinanceScoreP', 'ThirdPartyFinanceScore')
	PV('OngoingSupportScoreP', 'OngoingSupportScore')
	PV('MarketingScoreP', 'MarketingScore')

	#Pillar 4
	data['YrsBtwFranchise'] = data['FranchiseYearFr'] - data['FranchiseYear']
	data['Yrs Franchising'] = currentYear - data['FranchiseYearFr']
	data['YrsinBusiness'] = currentYear - data['FranchiseYear']

	data['UnitsPerYear'] = data['TotalUnits2016'] / data['Yrs Franchising']
	data['SocialScore'] = data['facebookFansNum'] + data['twitterFollowersNum']

	data = data.replace(np.inf, 0)
	PV('UnitsPerYearP', 'UnitsPerYear')
	PV('SocialScoreP', 'SocialScore')
	PV('TotalUnits2016P', 'TotalUnits2016')
	PV('YrsinBusinessP', 'YrsinBusiness')
	PV('Yrs FranchisingP', 'Yrs Franchising')
	data['SlowUnitGrowth_Penalty'] = data.apply(SlowUnitGrowthPenalty, axis=1)


	#Pillar 5
	data['CurrentRatio'] = data['CurrentAssets'] / data['CurrentLiabilities']
	data['DebtEquityRatio'] = (data['LongTermDebt'] + data['CurrentDebt'])/ data['TotalAssets']
	data['MemberEquityRatio'] = data['MemberEquity'] / data['TotalOperatingExpenses']
	data['ProfitMargin'] = data['NetIncome'] / data['GrossRevenue']
	data = data.replace(np.inf, 0)
	data = data.replace(-np.inf, 0)
	data = data.fillna(0)
	data['CurrentRatioScore'] = data.apply(FinScore_adj, args=('CurrentRatio',), axis=1)
	data['DebtEquityRatioScore'] = data.apply(FinScore_adj, args=('DebtEquityRatio',), axis=1)
	PV('MemberEquityRatioScore', 'MemberEquityRatio')
	PV('ProfitMarginScore', 'ProfitMargin')

	data['RecentFranchisePenalty'] = data.apply(RecentFranchisePenalty, axis=1)

	#Global Exclusion Rules
	data['ValidFranchise'] = data.apply(ValidFranchiseRules, axis=1)

	#Pillar Sub-Totals
	data['Pillar1'] = (data['FranchiseFeeP_Final']*FranchiseFee_w) + (data['Average FranchiseTotP']*AverageTotalInvestment_w) + ((1-data['FranchiseRoyaltyP'])*FranchiseRoyalty_w) + ((1-data['AdRoyaltyP'])*AdRoyalty_w) + ((1-data['TotalRoyaltyP'])*TotalRoyalty_w)
	data['Pillar2'] = (data['USUnits2016P']*USUnits_w) + (data['CanUts15P']*CanadaUnits_w) + (data['ForUts15P']*ForeignUnits_w) + (data['ComUnits15P']*CompanyUnits_w) + ((1-data['TerminationsP'])*Terminations_w) + (data['ReacquireP']*Reacquire_w) + (data['NonRenewP']*NonRenewal_w) + ((1-data['CeasedP'])*Ceased_w) + (data['AverageGrowthRateP']*AverageGrowthRate_w) + (data['AdjustedUnitsFinal']*AdjustedUnits_w)
	data['Pillar3'] = (data['TotalTrainingDaysP']*TotalTrainingDays_w) + (data['FranchiseEmployeePerUnitP']*FranchiseEmployee_w) + ((1-data['LitigationsPerUnitP'])*Litigations_w) + (data['InHouseFinanceScoreP']*InHouseFinance_w) + (data['ThirdPartyFinanceScoreP']*ThirdPartyFinance_w) + (data['OngoingSupportScoreP']*OngoingSupport_w) + (data['MarketingScoreP']*MarketingSupport_w)
	data['Pillar4'] = (data['UnitsPerYearP']*UnitsPerYear_w) + (data['SocialScoreP']*SocialScore_w) + (data['TotalUnits2016P']*Totalunits_w) + (data['YrsinBusinessP']*YrsinBusiness_w) + (data['Yrs FranchisingP']*YrsFranchising_w) - (data['SlowUnitGrowth_Penalty'])
	data['Pillar5'] = (data['CurrentRatioScore']*CurrentRatio_w) + (data['DebtEquityRatioScore']*DebtEquityRatio_w) + (data['MemberEquityRatioScore']*MemberEquityRatio_w) + (data['ProfitMarginScore']*ProfitMargin_w)
	data['subTotalScore'] = (data['Pillar1']*p1_w) + (data['Pillar2']*p2_w) + (data['Pillar3']*p3_w) + (data['Pillar4']*p4_w) + (data['Pillar5']*p5_w)

	data['RecentcyPenalty'] = np.multiply(data['RecentFranchisePenalty'],data.groupby('rankingCategory').std().loc[data.loc[:,'rankingCategory'],'subTotalScore'].replace(0,0.000001).replace(np.nan,0.000001))
	data['RecentcyPenalty'] = data['RecentcyPenalty'] / Penaltyweight
	data['FinalScore'] = data['subTotalScore'] - data['RecentcyPenalty'] - data['editorialScore']


	data.to_csv('F500-ErrorCheck.csv', encoding='utf-8')

def YNXrep (column_array):
	global data
	YNX_di = {'Y':1,'X':1,'N':0}
	for item in column_array:
		data = data.replace({item:YNX_di})

def Year_calc (row, column_array):
	for item in column_array:
		i = 0
		if row[item] == 0 & i == 0:
			return row[column_array[int(np.where(column_array==item)[0])-1]]
		i+=1
	return row[column_array[len(column_array)-1]]

def CPV(new_field, source_field, group_field):
	data[new_field] = scipy.stats.norm(data.groupby(group_field).mean().loc[data.loc[:,group_field],source_field], data.groupby(group_field).std().loc[data.loc[:,group_field],source_field].replace(0,0.000001).replace(np.nan,0.000001)).cdf(data[source_field])

def PV(new_field, source_field):
	data[new_field] = scipy.stats.norm(data[source_field].mean(), data[source_field].std()).cdf(data[source_field])

def FinScore_adj(row, field):
	if field == 'CurrentRatio':
		if row[field] <= .7:
			return Range_eval(row[field],0,.69,.0,.25)
		elif row[field] <= .9:
			return Range_eval(row[field],.7,.89,.25,.5)
		elif row[field] <= 1.1:
			return Range_eval(row[field],.9,1.10,.9,.95)
		elif row[field] <= 1.5:
			return Range_eval(row[field],1.1,1.49,.95,1)
		elif row[field] <= 3:
			return 1
		else:
			return Range_eval(row[field],3,data[field].max(),.95,1)
	elif field == 'DebtEquityRatio':
		if row[field] >= .5:
			return Range_eval(row[field],data[field].max(),.5,.7,0)
		elif row[field] >= .2:
			return Range_eval(row[field],.49,.2,.8,.9)
		else:
			return Range_eval(row[field],.2,0,.9,1)
	elif field == 'AdjustedUnitsP':
		return Range_eval(row[field],0,1,-1,1)

def Range_eval (val_in, Oldmin, Oldmax, Newmin, Newmax):
	return (((val_in - Oldmin) * (Newmax - Newmin)) / (Oldmax - Oldmin)) + Newmin

def AdjustedUnitGrowthClassification (row):
	if row['InvestmentVsCategoryAvg'] >= 1:
		if row['USGrowthVsCategoryAvg'] >= 1:
			return 1
		elif row['USGrowthVsCategoryAvg'] >= 0:
			return 2
		else:
			return 5
	elif row['InvestmentVsCategoryAvg'] >= 0:
		if row['USGrowthVsCategoryAvg'] >= 1:
			return 3
		elif row['USGrowthVsCategoryAvg'] >= 0:
			return 4
		else:
			return 6

def AdjustedUnitGrowthCalc(row):
	if row['AdjustedGrowthRank'] == 1:
		return (row['yoyUSGrowth']*(row['InvestmentVsCategoryAvg'] * 1.2))
	elif row['AdjustedGrowthRank'] == 2:
		return (row['yoyUSGrowth']*(row['InvestmentVsCategoryAvg'] * 1.1))
	elif row['AdjustedGrowthRank'] == 3:
		return (row['yoyUSGrowth']*(row['InvestmentVsCategoryAvg']))
	elif row['AdjustedGrowthRank'] == 4:
		return (row['yoyUSGrowth']*(row['InvestmentVsCategoryAvg']))
	elif row['AdjustedGrowthRank'] == 5:
		return (row['yoyUSGrowth']*(row['InvestmentVsCategoryAvg'] * 1.1))
	else:
		return (row['yoyUSGrowth']*((1+row['InvestmentVsCategoryAvg']) * 1.2))

def FranchiseFeePenalty (row):
	if row['FranchiseFee_Ratio'] >= FeeRatio_Penalty_percent:
		return (data['FranchiseFeeP'].std() * FeeRatio_Penalty_weight)
	else:
		return 0

def SlowUnitGrowthPenalty (row):
	if row['UnitsPerYear'] <= SlowGrowthPenaltyUnits:
		return (data['UnitsPerYearP'].std() * SlowGrowth_Penalty_weight)
	else:
		return 0

def ValidFranchiseRules (row):
	if row['TotalUnits2016'] > 9:
		if row['US Units 2016'] > 0:
			return 1
		elif row['CAN Units 2016'] > 0:
			return 1
	else:
		return 0

def RecentFranchisePenalty (row):
	if row['YrsBtwFranchise'] <= YrsBtwFranchise_threshold:
		if row['YrsinBusiness'] <= YrsinBusiness_threshold:
			return 6
		elif row['Yrs Franchising'] <= YrsFranchising_threshold:
			return 3
		else:
			return 0
	elif row['YrsinBusiness'] <= YrsinBusiness_threshold:
		return 5
	elif row['Yrs Franchising'] <= YrsFranchising_threshold:
		return 2
	else:
		return 0

def FiveYearGrowthBonus(row):
	if row['USNumberofYears'] >= 5:
		return (row['AverageGrowthRate'] + (row['AverageGrowthRate']/12))
	else:
		return row['AverageGrowthRate']

if __name__ == '__main__': main()
