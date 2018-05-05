#1.1 to 1.11 8.1 to 9.10  11.1 to 11.4  13.1 to 13.10  15.1-15.9  16.1  to 16.7
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
#boxplot and whisker(min max)
sns.boxplot(x='Species',y='Sepal.Length',data=iris_data)
plt.show()

#violin plot: it plot pdf along with boxplot
sns.violinplot(x='Species',y='Sepal.Length',data=iris_data)
plt.show()

#jointplot, multivariant density plot;2 dim pdf; contour
iris_setosa=iris_data[iris_data['Species']=='setosa']
sns.jointplot(x='Sepal.Length',y='Sepal.Length',data=iris_setosa,kind="kde")

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



#MINST dataset

