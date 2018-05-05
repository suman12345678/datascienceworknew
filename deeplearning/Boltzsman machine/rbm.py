#need to check again
# Boltzmann Machines for recommending movie 
#pytorch works in linux not on windows
# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
os.chdir('C:\\Users\\suman\\Desktop\\datasciencework\\deeplearning\\Boltzsman machine')

# Importing the dataset, encoding (as special char) = latin_1
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set, total 5 cross validate training set are available
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies from training and set test
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
#each list for one user , each line for one user and each column for all movies similar to TDM
#if user didno rate make it 0
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]#movies which are rated by that user
        id_ratings = data[:,2][data[:,0] == id_users]#get all the ratings for that user
        ratings = np.zeros(nb_movies)#initialize to 0
        ratings[id_movies - 1] = id_ratings#replace 0 with actual rating if actually given
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
#so here input nodes are all movies, each user is one obs

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#till now common preprocessing for recommendation system is done


# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1#not rated by user
training_set[training_set == 1] = 0#not liked
training_set[training_set == 2] = 0#not liked
training_set[training_set >= 3] = 1#liked
test_set[test_set == 0] = -1#not rated by user
test_set[test_set == 1] = 0#not liked
test_set[test_set == 2] = 0#liked
test_set[test_set >= 3] = 1#liked

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):#visible node,hidden node
        self.W = torch.randn(nh, nv)#init weight, probability of visible node given hidden node
        self.a = torch.randn(1, nh)#init bias, prob of hidden node given visible node
        self.b = torch.randn(1, nv)#init bias, prob of visible node given hidden node
    def sample_h(self, x):#probability of hidden node given visible node p(h) given v. 
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)#prob of hidden node activated given value of visible node
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):#probability of visible node given hidden node
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)#we have 1682 movies=1682 visible node
        return p_v_given_h, torch.bernoulli(p_v_given_h)# so we have 1682 probability
    def train(self, v0, vk, ph0, phk):#constractive vivergence: gibbs sampling i.e. approximating log likelihood gradient
    #.maximizing log likelihood i.e. minimising energy
    #based on v0 we sample hiddennode h1 in first iteration,then with this input h1 we find sample of v i.e. v1
    #this is done k times, so here we update w,b,a
    #v0 ratings of all movies for one user,vk=visible node,ph0:P(hiddennode)=1 given 
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)#ph0 is prob of hidden node=1 given v0,phk in kth iteration
        self.b += torch.sum((v0 - vk), 0)#
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])#no of movies or visible node
nh = 100#no of hidden node(features like oscar,gener,actor..) lets make it 100
batch_size = 100#update weights after 100 obs
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10#total obs is looked 10 times
for epoch in range(1, nb_epoch + 1):
    train_loss = 0#measure diff bet predicting and real rating
    s = 0.#counter, increment after each epoch to normalise error
    for id_user in range(0, nb_users - batch_size, batch_size):#batch learning 100 users at a time
        vk = training_set[id_user:id_user+batch_size]#initial vk as same as v0 at first
        v0 = training_set[id_user:id_user+batch_size]#original rating keep it to compare
        ph0,_ = rbm.sample_h(v0)#get only first element of function, p of hidden node=1 given visible node
        for k in range(10):#k step of constractive divergence,mcmc technique
            _,hk = rbm.sample_h(vk)#update hk,sammple hidden node
            _,vk = rbm.sample_v(hk)#update vk ,sammple visible node
            vk[v0<0] = v0[v0<0]#remove where rating -1(no rating)
        #get last sample of visible node    
        phk,_ = rbm.sample_h(vk)#get phk also as input to train method
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM similar like training
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]#input,trainig set used to activate neuron
    vt = test_set[id_user:id_user+1]#target
    if len(vt[vt>=0]) > 0:#we need only one step of random walk
    #dont consider non eistance rating
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))