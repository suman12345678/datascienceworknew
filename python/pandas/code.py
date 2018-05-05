# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:15:46 2018

@author: suman
"""

#put one file at a time in the folder and run script it will update master
#togetger 2 file not working
import os
os.chdir("C:\\Users\\ba4205\\Desktop\\batch documantation\\code in out")
import pandas as pd
#read master
#take 1st 2 rows seperately
first2=pd.read_excel('master.xlsx',sheet='To be documented').iloc[:2,]
first=first2.columns
second=first2.iloc[0:2,]
a=pd.read_excel('master.xlsx',sheet='To be documented',skiprows=2)
#4242 row 22 col
a.describe()#27913 row 22 col
#remove junk rows
a=a[a["JOBNAME"].notnull()]
#check different values in BRID
#a.groupby(['BRID']).size()
#check already updated rows
a1=a[a["BRID"].notnull()]
#a1=a.drop(a.index[[0,1]])
import re
path="C:\\Users\\ba4205\\Desktop\\batch documantation\\code in out"
inputfile=[]
#for filename in glob.glob(os.path.join(path, '*.xlsx')):
for filename in os.listdir(os.getcwd()):
  if (filename=='master.xlsx' or filename=='output.xlsx'):
      print("left input and outputfile")
  elif (re.search(r'^[^~$]+.xlsx',filename)):
      print(filename)
      inputfile.append(filename)
  else:
      print("left file",filename)

print("total no of child file",len(inputfile),inputfile)
#test for only one file later remove below line
#inputfile=['input.xlsx']
#test uncomment later
for l in inputfile:
    print("enter userid for file",l) #test with BC8637
    filenamein=input('enter:')
    b=pd.read_excel(l,sheet='To be documented',skiprows=2) # row 22 col
    #compare master and inputs
    print("no of rows in master and child",len(a.index),len(b.index))
    #input("want to continue?")
    in1=input('want to continue y/n?')
    if in1=='n':
        break
    #remove junk rows
    #print("no exec")
    b=b[b["JOBNAME"].notnull()]
    #b.iloc[[1],[11]]
    #b1=b.drop(b.index[[0,1]])
#b.iloc[25237,11]
#take the updated ones from user BC8637
    b1=b[b["BRID"] ==filenamein]#49
    print("no of updated rows by user id",len(b1.index))#79
    print("no of already updated rows in master",len(a1.index))#4242
    with open("log.txt","a") as logfile:
        logfile.write("\nno of already rows in master "+str(len(a1.index)))
        logfile.write("\nno of updated rows by user id "+filenamein+"
"+str(len(b1.index)))
        logfile.write("\ntotal master will be
"+str(len(b1.index)+len(a1.index)))
    logfile.close()
    #print("rows", b1['JOBNAME'])
    merg = a.merge(b1, left_on=['JOBNAME','ADRID'],
right_on=['JOBNAME','ADRID'], how='left')
    merg.loc[merg['DESCRIPTION_y'].notnull(), 'DESCRIPTION_x'] =
merg['DESCRIPTION_y']
    merg=merg.drop('DESCRIPTION_y',1)
    merg.loc[merg['BUSINESS_IMPACT_y'].notnull(), 'BUSINESS_IMPACT_x']
= merg['BUSINESS_IMPACT_y']
    merg=merg.drop('BUSINESS_IMPACT_y',1)
    merg.loc[merg['PROCESS_IMPACT_y'].notnull(), 'PROCESS_IMPACT_x'] =
merg['PROCESS_IMPACT_y']
    merg=merg.drop('PROCESS_IMPACT_y',1)
    merg.loc[merg['RERUN_GUIDANCE_y'].notnull(), 'RERUN_GUIDANCE_x'] =
merg['RERUN_GUIDANCE_y']
    merg=merg.drop('RERUN_GUIDANCE_y',1)
    merg.loc[merg['EXT_PART_DELIVERY_y'].notnull(),
'EXT_PART_DELIVERY_x'] = merg['EXT_PART_DELIVERY_y']
    merg=merg.drop('EXT_PART_DELIVERY_y',1)
    merg.loc[merg['PLATFORM_REQUIREMENT_y'].notnull(),
'PLATFORM_REQUIREMENT_x'] = merg['PLATFORM_REQUIREMENT_y']
    merg=merg.drop('PLATFORM_REQUIREMENT_y',1)
    merg.loc[merg['CRITICAL_DEADLINE_y'].notnull(),
'CRITICAL_DEADLINE_x'] = merg['CRITICAL_DEADLINE_y']
    merg=merg.drop('CRITICAL_DEADLINE_y',1)
    merg.loc[merg['MANUAL_EXECUTION_y'].notnull(),
'MANUAL_EXECUTION_x'] = merg['MANUAL_EXECUTION_y']
    merg=merg.drop('MANUAL_EXECUTION_y',1)
    merg.loc[merg['BRID_y'].notnull(), 'BRID_x'] = merg['BRID_y']
    merg=merg.drop('BRID_y',1)
    merg=merg.drop(['CRITICALITY_y','CRITICAL_PATH_y','MTTS_y','IN_PROGRESS_y','SPI_y','SUBSPI_y','MAIN_AREA_y','AREA_y','DEPARTMENT_y','PROJECT_y','ADRWSID_y'],1)
#rename columns
    #merg.columns=['JOBNAME','DESCRIPTION','BUSINESS_IMPACT','PROCESS_IMPACT','RERUN_GUIDANCE','EXT_PART_DELIVERY','PLATFORM_REQUIREMENT','CRITICALITY','CRITICAL_DEADLINE','CRITICAL_PATH','MANUAL_EXECUTION','BRID','MTTS','IN_PROGRESS','SPI','SUBSPI','MAIN_AREA','AREA','DEPARTMENT','PROJECT','ADRID','ADRWSID']
    print("new rows in master",len(merg.index))#4321
#add 2 lines
    #c=['8 Char','up to 500 char','Up to 250 Char','Up to 250
Char','Up to 250 Char/
    #Drop down list - only the 4 listed values must be used" Up to 100
Char "Up to 50 Char/
    #Drop down list - Only the listed values must be used" Dont write
in this column 4 Char Dont write in this column 1 Char 8 Char Dont
write in this column Don't write in this column Don't write in this
column Don't write in this column Don't write in this column Don't
write in this column Don't write in this column Don't write in this
column Don't write in this column Don't write in this column
    #]
    merg.columns=first
    #add first 2 lines and change column name
    merg=second.append(merg)
    merg=merg.reset_index()
    #removed created column index
    merg=merg.drop('index',axis=1)

    merg.to_excel("output.xlsx",index=False,sheet_name='To be documented')
#delete and rename existing files
    os.remove("master.xlsx")
    os.rename("output.xlsx","master.xlsx")

#import glob

