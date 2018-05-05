# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:29:18 2017

@author: suman
"""


#calculate average of a field in excel in 2 decimal
import csv
import os
#%precision 2
os.chdir("C:\\Users\\suman\\Desktop\\datasciencework\\python\\courseratraining")

'''
with open("iris.csv") as f:
    a=list(csv.DictReader(f))
print(sum(float(i['Sepal.Length'] ) for i in a))
    

#check unique species 
print(set(i['Species'] for i in a))


#pandas operation
data=pd.read_csv("iris.csv")
a=data.loc[0:3,'Sepal.Length']
data.iloc[0:3,1]
a=data.iloc[0:3,1].copy()#this will not change data even a is changed

#return a rowname where a colvalue is max
''.join(list(df[(df['Col2']==df['Col2'].max())].index))

#which row has bigges difference between col1 and col2
''.join(list(df[(df['Col1']-df['Col2'])==(df['Col1']-df['Col2']).max()].index))

#rario of col1-col2/col3 difference
df1=df[(df['Col1']>0) & (df['Col2']>0)]
   return ''.join(list(df1[((df1['Col1']-df1['Col2'])/df1['Col3'])==((df1['Col1']-df1['Col2'])/df1['Col3']).max()].index))
    
    
 #calculation of columns of df and make series
a=df['col1']*3+df['col2']*2+df['col3']

#group by a column and maximum
a=df[df['col1]==50]
b=a.groupby('col2').count()['col1'].idxmax()  

#Only looking at the three most populous counties for each state, what are the three most populous states (in order of highest population to lowest population)? Use CENSUS2010POP.
a = census_df[census_df['SUMLEV'] == 50]
    b = a.sort_values(by=['STNAME','CENSUS2010POP'],ascending=False).groupby('STNAME').head(3)
    c = b.groupby('STNAME').sum().sort_values(by='CENSUS2010POP').head(3).index.tolist()
    
def answer_five():
    a=census_df[census_df['SUMLEV']==50]
    b=a.groupby('STNAME').count()['SUMLEV'].idxmax()
    print(type(b))
    return b
answer_five()

def answer_six():
    a = census_df[census_df['SUMLEV'] == 50]
    b = a.sort_values(by=['STNAME','CENSUS2010POP'],ascending=False).groupby('STNAME').head(3)
    c = b.groupby('STNAME').sum().sort_values(by='CENSUS2010POP').head(3).index.tolist()
    print(type(c))
    return c
answer_six()

def answer_seven():
    pop = census_df[['STNAME','CTYNAME','POPESTIMATE2015','POPESTIMATE2014','POPESTIMATE2013','POPESTIMATE2012','POPESTIMATE2011','POPESTIMATE2010']]
    pop = pop[pop['STNAME']!=pop['CTYNAME']]
    index = (pop.max(axis=1)-pop.min(axis=1)).argmax()
    return census_df.loc[index]['CTYNAME']

answer_seven()

def answer_eight():
    counties_df = census_df[census_df['SUMLEV'] == 50]
    ans = counties_df[((counties_df['REGION']==1)|(counties_df['REGION']==2))&(counties_df['CTYNAME']=='Washington County')&(counties_df['POPESTIMATE2015']>counties_df['POPESTIMATE2014'])][['STNAME','CTYNAME']]
    #print(type(ans))
    return ans
answer_eight()
'''
#assignment 3
#=============
#**************
import pandas as pd
import numpy as np
x = pd.ExcelFile('Energy Indicators.xls')
#skip row
energy = x.parse(skiprows=17,skip_footer=(38))
#name the column
energy = energy[['Unnamed: 1','Petajoules','Gigajoules','%']]
#rename coulmn
energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
#remove na value
energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] =  energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
energy['Energy Supply'] = energy['Energy Supply']*1000000
#replace HIng Kong.. to China
energy['Country'] = energy['Country'].replace({'China, Hong Kong Special Administrative Region':'Hong Kong','United Kingdom of Great Britain and Northern Ireland':'United Kingdom','Republic of Korea':'South Korea','United States of America':'United States','Iran (Islamic Republic of)':'Iran'})
#replace Chine(a country of great wall) to Chine
energy['Country'] = energy['Country'].str.replace(r" \(.*\)","")


pd.read_csv('world_bank.csv',skiprows=4,skip_footer=(1))
#change country namw from korea rep ro south korea
GDP['Country Name'] = GDP['Country Name'].replace('Korea, Rep.','South Korea')
GDP['Country Name'] = GDP['Country Name'].replace('Iran, Islamic Rep.','Iran')
GDP['Country Name'] = GDP['Country Name'].replace('Hong Kong SAR, China','Hong Kong')
#select few columns
GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
#rename columns
GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']


ScimEn = pd.read_excel(io='scimagojr.xlsx')
ScimEn_m = ScimEn[:15]

    
#merge dataframes
df = pd.merge(ScimEn_m,energy,how='inner',left_on='Country',right_on='Country')
final_df = pd.merge(df,GDP,how='inner',left_on='Country',right_on='Country')
#set Country as index to be appeared
final_df = final_df.set_index('Country')    

#ven diagram to know how many loose answer 2
# Union A, B, C - Intersection A, B, C
union = pd.merge(pd.merge(energy, GDP, on='Country', how='outer'), 
    ScimEn, on='Country', how='outer')
intersect = pd.merge(pd.merge(energy, GDP, on='Country'), 
    ScimEn, on='Country')
len(union)-len(intersect)


#avg GDP for each country for last 10 years answer 3
Top15 = answer_one()
years = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
(Top15[years].mean(axis=1)).sort_values(ascending=False).rename('avgGDP')

#answer 4 By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
Top15 = answer_one()
Top15['avgGDP'] = answer_three()
Top15.sort_values(by='avgGDP', inplace=True, ascending=False)
abs(Top15.iloc[5]['2015']-Top15.iloc[5]['2006'])

#answer 5 What is the mean Energy Supply per Capita?
Top15 = answer_one()
print()
Top15['Energy Supply per Capita'].mean()


#answer  6 What country has the maximum % Renewable and what is the percentage?
Top15 = answer_one()
    ct = Top15.sort_values(by='% Renewable', ascending=False).iloc[0]
    return (ct.name, ct['% Renewable'])

#answer 7 Create a new column that is the ratio of Self-Citations to Total Citations. What is the maximum value for this new column, and what country has the highest ratio?
Top15 = answer_one()
Top15['Citation_ratio'] = Top15['Self-citations']/Top15['Citations']
ct = Top15.sort_values(by='Citation_ratio', ascending=False).iloc[0]
return ct.name, ct['Citation_ratio']

#answer 8 Create a column that estimates the population using Energy Supply and Energy Supply per capita. What is the third most populous country according to this estimate?
Top15 = answer_one()
Top15['Population'] = Top15['Energy Supply']/Top15['Energy Supply per Capita']
Top15.sort_values(by='Population', ascending=False).iloc[2].name

#answer 9 Create a column that estimates the number of citable documents per person. What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the .corr() method, (Pearson's correlation).
Top15 = answer_one()
Top15['Estimate Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
Top15['avgCiteDocPerPerson'] = Top15['Citable documents'] / Top15['Estimate Population']
Top15[['Energy Supply per Capita', 'avgCiteDocPerPerson']].corr().ix['Energy Supply per Capita', 'avgCiteDocPerPerson']

def plot9():
    import matplotlib as plt
    %matplotlib inline
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])
    
#answer 10 Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
Top15 = answer_one()
mid = Top15['% Renewable'].median()
Top15['HighRenew'] = Top15['% Renewable']>=mid
Top15['HighRenew'] = Top15['HighRenew'].apply(lambda x:1 if x else 0)
Top15.sort_values(by='Rank', inplace=True)
Top15['HighRenew']

#answer 11 Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.

#ContinentDict  = {'China':'Asia', 
#                  'United States':'North America', 
#                  'Japan':'Asia', 
Top15 = answer_one()
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    groups = pd.DataFrame(columns = ['size', 'sum', 'mean', 'std'])
    Top15['Estimate Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    for group, frame in Top15.groupby(ContinentDict):
        groups.loc[group] = [len(frame), frame['Estimate Population'].sum(),frame['Estimate Population'].mean(),frame['Estimate Population'].std()]
    return groups

#Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
Top15 = answer_one()
Top15['PopEst'] = (Top15['Energy Supply'] / Top15['Energy Supply per Capita']).astype(float)
Top15['PopEst'].apply(lambda x: '{0:,}'.format(x))

#assignment 2 plotting
#plot of line graph of weather report
"""  ID 	       Date 	       Element Data_Value
0 	IDM00096087 	2011-07-08 	TMIN 	256
1 	IDM00096087 	2015-03-06 	TMIN 	234
2 	IDM00096091 	2015-03-08 	TMIN 	243
3 	SNM00048698 	2006-04-18 	TMAX 	324
4 	IDM00096179 	2015-11-21 	TMAX 	318 """
df = pd.read_csv('data/C2A2_data/BinnedCsvs_d100/4e86d2106d0566c6ad9843d882e72791333b08be3d647dcae4f4b110.csv')
#sort on ID and Date
df.sort(['ID','Date']).head()
#create 2 variable year an dmonth
df['Year'],df['Month-Date']=zip(*df['Date'].apply(lambda x:(x[:4],x[5:])))
#remove rows with 29feb
df = df[df['Month-Date'] != '02-29']
#find TMIN and TMAX for !2015 group by Month-Date, i.e. for each day of year min and max
import numpy as np
temp_min=df[(df['Element']=='TMIN') & (df['Year']!='2015')].groupby('Month_Date').aggregate({'Date_value':np.min})
#only for 2015
temp_min_15 = df[(df['Element'] == 'TMIN') & (df['Year'] == '2015')].groupby('Month-Date').aggregate({'Data_Value':np.min})
#check whose where 2015 data broke the recorfs
broken_min = np.where(temp_min_15['Data_Value'] < temp_min['Data_Value'])[0]
#plot high and low and point where 2015 broke the records
plt.figure()
plt.plot(temp_min.values, 'b', label = 'record low')
plt.plot(temp_max.values, 'r', label = 'record high')
plt.scatter(broken_min, temp_min_15.iloc[broken_min], s = 10, c = 'g', label = 'broken low')
plt.scatter(broken_max, temp_max_15.iloc[broken_max], s = 10, c = 'm', label = 'broken high')
plt.gca().axis([-5, 370, -150, 650])
plt.xticks(range(0, len(temp_min), 20), temp_min.index[range(0, len(temp_min), 20)], rotation = '45')
plt.xlabel('Day of the Year')
plt.ylabel('Temperature (Tenths of Degrees C)')
plt.title('Temperature Summary Plot near Singapore')
plt.legend(loc = 4, frameon = False)
#make shade
plt.gca().fill_between(range(len(temp_min)), temp_min['Data_Value'], temp_max['Data_Value'], facecolor = 'yellow', alpha = 0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


#custom visualization
#====================
#use following data for assignment
%matplotlib notebook
import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])

from scipy import stats
year_avg = df.mean(axis = 1)
year_std = df.std(axis = 1)
#maybe standard error
yerr = year_std / np.sqrt(df.shape[1]) * stats.t.ppf(1-0.05/2, df.shape[1]-1)
import matplotlib.pyplot as plt
plt.figure()
plt.show()
bars = plt.bar(range(df.shape[0]), year_avg, yerr= yerr, alpha = 0.6, color = 'rgby')
threshold=42000
plt.axhline(y = threshold, color = 'grey', alpha = 1)

plt.xticks(range(df.shape[0]), ['1992', '1993', '1994', '1995'], alpha = 0.8)
plt.title('Ferreira et al, 2014')



#project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import os
%matplotlib notebook
plt.style.use('seaborn-colorblind')
df = pd.read_csv('data/C2A2_data/BinnedCsvs_d25/9bc594d0d6bf5fec16beb2afb02a3b859b7d804548c77d614b2a6b9b.csv')
df2 = pd.read_csv('https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv')
df=df.sort(['ID', 'Date'])
df2=df2.set_index('Date')
df2=df2[df2['Source']=='GCAG']
df2=df2.reset_index()
df2['Year']=df2['Date'].apply(lambda x:x[:4])
df2['Month']=df2['Date'].apply(lambda x:x[5:])
df2
