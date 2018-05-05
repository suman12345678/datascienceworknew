# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:12:25 2017

@author: suman
"""

import os
os.chdir("C:\\Users\\suman\\Desktop\\datasciencework\\python\\iris")
import pandas as pd
iris_data=pd.read_csv("iris.csv")
#as missing values are marked as NA mark NA values as missing in pandas
iris_data=pd.read_csv("iris.csv",na_values=['NA'])
print(iris_data.head())
print(iris_data.describe())
print(iris_data.shape)
print(iris_data.columns)
print(iris_data['Species].count_values())

#scatter
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
sns.FacetGrid(iris_data,hue="Species",size=4) \
 .map(plt.scatter,"Sepal.Length","Sepal.Width").add_legend()


#pairplot difficult for many feature(use pca tsne)
sns.set_style("whitegrid")
sns.pairplot(iris_data,hue="Species",size=4) 

#histogram
sns.set_style("whitegrid")
sns.FacetGrid(iris_data,hue="Species",size=6) \
 .map(sns.distplot,"Sepal.Length").add_legend()
plt.show()

#25th quantile=X 25% of the data value are less than X
#median abs deviation  IQR=75%value -25% value


#check 2nd row and 2ndrow,1column
print(iris_data.loc[1]) #its a seris
type(iris_data.loc[1])
iris_data.loc[1][0]#its numpy
#check one column
iris_data['Species']#its type is series

#plot data
import matplotlib.pyplot as plt
import seaborn as sb
sb.pairplot(iris_data.dropna(),hue='Species')

#check if any na value exist
iris_data.loc[(iris_data['Sepal.Length'].isnull()) |              
              (iris_data['Sepal.Width'].isnull()) |
              (iris_data['Petal.Length'].isnull()) |
              (iris_data['Petal.Width'].isnull())]


