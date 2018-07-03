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

#####after payment
#4.1-4.10
import pdb
def seq(n):
    for i in range(n):
        pdb.set_trace() #breakpoint 
        print(i)
    return

#12
#generate random sample of 30 out of 150
p=30/150;
from sklearn import datasets
import random
iris=datasets.load_iris()
d=iris.data
sample_data=[]
for i in range(0,150):
    if (random.random() <= p): #any point to be picked has a probability = p
      sample_data.append(d[i,:])
print(len(sample_data))      
        
#convert pareto to gausian by boxcox
from scipy import stats
import matplotlib.pyplot as plt
#plot a non-normal disturibution by simulation
fig=plt.figure()
ax1=fig.add_subplot(211)
x=stats.loggamma.rvs(5,size=500)+5
prob=stats.probplot(x,dist=stats.norm,plot=ax1)
ax1.set_xlabel('')
ax1.set_title("prob plot against normal dist")
#now apply boxplot to convert to normal dist
fig=plt.figure()
x=stats.loggamma.rvs(5,size=500)+5
ax2=fig.add_subplot(211)
xt,_=stats.boxcox(x)
prob=stats.probplot(xt,dist=stats.norm,plot=ax2)
ax2.set_title("prob plot after boxcox")


#calculate median with 95% CI using bootstrap
from sklearn.utils import resample
import numpy as np
from matplotlib import pyplot
x=np.array([180,162,158,172,168,150,171,183,165,176])
n=1000
size=int(len(x))
median=list()
for i in range(n):
    s=resample(x,n_samples=size)
    m=np.median(s)
    median.append(m)
    
pyplot.hist(median)
pyplot.show()
alpha=.95
p=((1-alpha)/2)*100
lower=np.percentile(median,p)

p=(alpha+((1-alpha)/2))*100
upper=np.percentile(median,p)
print('ci',alpha,lower,upper)


#k-s test
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
#generate gaussian
x=stats.norm.rvs(size=1000)
sns.set_style('whitegrid')
sns.kdeplot(np.array(x),bw=.5)
plt.show()
#now compare this with a norm dist
stats.kstest(x,'norm')
#we are getting a p value .69 so null hypothesis is true i.e. its from normal dist
#now try k-s test with uniform dist
y=np.random.uniform(0,1,10000)
sns.kdeplot(np.array(y),bw=.1)
plt.show()
stats.kstest(y,'norm')
#here p value=0 and D value .5 as P value is 0 null hypothesis rejected



    