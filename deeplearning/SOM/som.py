# Self Organizing Map
#detect credit card application fraud by unsupervised
#each customer(15 attribute) will be assigned to a output node(winning) which is closeest
#then each neighbour node's weight within radius is also updated, closer more update of weight 
#this reduction of output space reduces in each iteration until no more reduction
#and ultimately some winning output neuron established
#MID=mean inter neuron distance is done for each output neuron to its neighbour neurons
#then we can find the outliner neurons and later customer associated with those neuron are selected as outliner
#higher the MID the more winning node is far from its neighbour node so its outliner
# so the winning node with highest MIDs we have to take as fraud
# we will use color the largest MID color close to white
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("C:\\Users\\suman\\Desktop\\datasciencework\\deeplearning\\SOM")
# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
#split into 2 X= all columns and Y= whether application was accepted or rejected
# we dont need y here as its unsupervised, by seeing X we want to create SOM
#y=0/1 doesnot mean fraud nor not, this is application rejected or accepted
#we have to find out whether customer is actually fraud or not

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
#minisom code is put into the same directory to use it
from minisom import MiniSom
#x=10 and y=10 mean 10*10 grid of nuron(cluster), input_len=features in X,sigma=raidus of neighbourhood grid
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
#randomly initialize weight of neurons
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
#get 10*10 node with MID using color, more white more MID
pcolor(som.distance_map().T)
#outliners are having highest mid(distance from node)
colorbar()
#circle and square(circle for not getting approval,square for getting approval)
markers = ['o', 's']
#red who didnot get approval g who got approval
colors = ['r', 'g']
#keep the customer near winning node
for i, x in enumerate(X):
    #get winning node of customer
    w = som.winner(x)
    #if the customer get approval make the winning node green square, else red circle
    plot(w[0] + 0.5,#x coordinate of winning node
         w[1] + 0.5,#y coordinate of winning node
         markers[y[i]],
         markeredgecolor = colors[y[i]],#if cust get approval g, else r
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
#if we look map and in outliner(white box) if any green sqaure are there
#that means high risk as those customer got approval 
# Finding the frauds by reverse mapping
mappings = som.win_map(X)#get dictionary of winning node coordinate and customer
#get the coordinate of winning node of color white (8,1) and (6,8)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
#now in this list some are approved action will be taken on those
  