# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 23:56:04 2018

@author: suman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:36:32 2018
@author: ba4205
"""
# Recurrent Neural Network

# Part 1 - Data Preprocessing
# Importing the libraries
#import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#os.chdir('C:\\Users\\suman\\Desktop\\datasciencework\\deeplearning\\RNN')
# Importing the training set
import pyodbc
connection = pyodbc.connect('Driver={SQL Server};'
                                'Server=C44S03\INST003;'
                                'Database=ITSM;'
                                'Trusted_Connection=yes')
cursor = connection.cursor()
SQLCommand = ("SELECT cast(open_time as date) as date,count(*) as
count FROM dbo.Incidents where Assignment_Group like 'F161%' or
Assignment_Group like 'F162%' or Assignment_Group like 'F163%'  or
Assignment_Group like 'F164%' or Assignment_Group like 'F182%'  and
OPEN_TIME>'2017-01-15 09:34:00.000' group by cast(open_time as date)
order by cast(open_time as date)")
data = pd.read_sql(SQLCommand, connection)

connection.close()
#plot historic data
plt.plot(data.iloc[:,0:1].values, data.iloc[:,1:2].values, color =
'red', label = 'historic incident')
plt.title('Incident data for DWH & AML')
plt.xlabel('date')
plt.ylabel('Total incident')
#plt.xlim(1,200)
plt.legend()
plt.show()
#first use training data till 31-12-2017 and test data from 01-01-2018
#len(data.index),
dataset_train = data[data['date']<='2017-12-31']
dataset_test = data[data['date']>'2017-12-31']
#take only open count field as training
training_set = dataset_train.iloc[:, 1:2].values
# Feature Scaling: standarization(x-mean(x)/sd(x)) and
normalization(x-min(x)/max(x)-min(x))
#for rnn better use normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
#everytime it will check last timestamp past memory of 60 data. 60 is with trial
X_train = []
y_train = []
for i in range(60, len(dataset_train.index)):#it can start from 60,
len(dataset_train.index) is the total no in training data
    X_train.append(training_set_scaled[i-60:i, 0])#append 60 previous
count so here x_train is 60 less than actual
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
#so here X_train has 60 feature(old data before ith day) and y_train
ha s1 feature(ith day data)
# Reshaping by adding 1 dimension(from keras)[row,col,timestep=1]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

# Part 2 - Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense#fully connected
from keras.layers import LSTM
from keras.layers import Dropout#dropout regularization to handle
overfitting drop some neuron
# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
#units=no of LSTM cell should be high as stock data,return_seq =
feedback to previous neuron,
#input_shape=in 3 dim but only last 2 dim required (60,1)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape =
(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation, here
input is not required
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation as the
last layer rerun is not required
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#from keras.layers import Flatten
#regressor.add(Flatten())
# Adding the output layer here only one output so units=1(one neuron
as regression), and Dense for fully connected layer
regressor.add(Dense(units = 1))
# Compiling the RNN, 'adam' for  from keras documentation always safe choice,
# 'mean_squared_error' as its a regression problem and not
classification problem
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set, epochs=100 time whole data is propagated,
#batch_size=after 32 obs update weight insteda of doing it for each obs
#in each epoch loss is reduced
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Part 3 - Making the predictions and visualising the results
# Getting the real count of 2018
#dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_count = dataset_test.iloc[:, 1:2].values



# Getting the predicted count of 2018
#we need 60 previous date, so we need both training and test set,also
need to scale as rnn is trained on scaled of train set
dataset_total = pd.concat((dataset_train['count'],
dataset_test['count']), axis = 0)
#make numpy array
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#reshape numpy
inputs = inputs.reshape(-1,1)
#scale test as training
inputs = sc.transform(inputs)
#create test like train
X_test = []
#prediction for januart 2018 i.e. 20 days so 60 to 80
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
#create 3d format like traing
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_count = regressor.predict(X_test)
#inverse transform as it was normalize
predicted_count = sc.inverse_transform(predicted_count)
# Visualising the results,here predicted will show only a trend not exact
#model cannot get nonlinear change, it reacts with smooth change
plt.plot(real_count, color = 'red', label = 'Real count')
plt.plot(predicted_count, color = 'blue', label = 'Predicted count')
plt.title('DWH n AML count incident')
plt.xlabel('Time')
plt.ylabel('count')
plt.legend()
plt.show()


# Getting the predicted count of next day
#we need 60 previous date, so we need both training and test set,also
need to scale as rnn is trained on scaled of train set
dataset_total = pd.concat((dataset_train['count'],
dataset_test['count']), axis = 0)
#make numpy array
inputs = dataset_total[len(dataset_total) - 60:].values
#reshape numpy
inputs = inputs.reshape(-1,1)
#scale test as training
inputs = sc.transform(inputs)
#create test like train



X_test = []
#prediction for next  day
for i in range(60, 61):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
#create 3d format like traing
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_count = regressor.predict(X_test)
#inverse transform as it was normalize
predicted_count = sc.inverse_transform(predicted_count)


#prediction for next 20 day need to check
X_test = []#list
#a=[]
inputs_temp=inputs#numpy.ndarray 56: => array([[0.16049383],
[0.16049383], [0.0617284 ],[0.08641975]])
for i in range(60, 70):
    X_test.append(inputs_temp[i-60:i, 0])#[array([0. ,..., 0.16049383,
0.0617284 , 0.08641975])]
    #X_test[0][1] = .012345679012345678
    X_test1 = np.array(X_test)
    #create 3d format like traing
    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
    predicted_count = regressor.predict(X_test1)#numpy array =>
array([[0.04815132]], dtype=float32)
    #a.append(predicted_count)
    #add predicted element in inputs
    #print(len(inputs_temp),len(X_test1[i-60]),predicted_count)
    inputs_temp=np.concatenate((inputs_temp,predicted_count[[i-60]]))
    #X_test = []
#inverse transform as it was normalize
#predicted_count = regressor.predict(X_test1)
predicted_count = sc.inverse_transform(predicted_count)
#a=sc.inverse_transform(a)
#plt.plot(real_count, color = 'red', label = 'Real count')
plt.plot(range(predicted_count.shape[0]),predicted_count ,color =
'blue', label = 'Predicted count')
plt.title('DWH n AML count incident prediction for next 20 days')
plt.xlabel('Date')
plt.ylabel('count')
plt.legend()
plt.show()