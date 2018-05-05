__author__="suman"
import pandas as pd
import matplotlib.pyplot as plot 
from pandas import DataFrame
raw_data=pd.read_table("sonar.all-data.txt",sep=",",header=None)  
raw_data.shape
raw_data.describe(include='all')
#raw_data.columns=['Sex', 'Length', 'Diameter', 'Height','Whole weight','Shucked weight', 'Viscera weight',                   'Shell weight', 'Rings']

#plot M and F
for i in range(raw_data.shape[0]):
  if raw_data.iat[i,60]=="R":
      pcolor="red"
  else:
      pcolor="blue"
  a=raw_data.iloc[i,0:60]
  a.plot(color=pcolor)
  
plot.xlabel("attribute index")
plot.ylabel("attr value")
plot.show()

#check corelation of variables
DataFrame(raw_data.iloc[:,1:60].corr())
