# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:43:34 2018

@author: suman
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir('C:\\Users\\suman\\Desktop\\deep learning\\Artificial_Neural_Networks')

# Importing the dataset and seperate input and output variable
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values
# Encoding 2 categorical data into numeric fo processing into ANN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#create dummy variable for country as there are 3 catagory for sex its not required as only 2 catagory
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#now as there are 3 dummy variable created for country remove one(1st one) as 2 are enogh
X=X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling i.e. normalizing the data for processing in ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Create classifier here
import keras
from keras.models import Sequential
from keras.layers import Dense
#use dropout regularization to improve overfitting and stopping neuron to learn too much
# by randomly disableing some neurons
from keras.layers import Dropout
#initializing ann, below classifier is the model initial stage
classifier=Sequential()
#adding input layer and first hidden layer. For 
#for hidden layer selection done by : avg of input layer n output layer=6, relu=rectifier
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
# with dropout of 10% of neurons in 1st hidden layer so randomly 10% neuron will be disable to improve overfitting
classifier.add(Dropout(p=.1))
#create second layer here input_dim not required as its already know
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# with dropout of 10% of neurons in 2nd hidden layer
classifier.add(Dropout(p=.1))
#add output layer, change output_dim should be 1 as only one node, acd activation func should be sigmoid to get probability
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#for more than 2 catagory activation will be softmax and output_dim=3
#compile ann 
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting ann to training set, this will show accuracy and loop through, batch_size i.e. after 32 row weights will be updated
#epoch =10 i.e. all obs will be lop 10 times
classifier.fit(X_train,y_train,batch_size=32,nb_epoch=10)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#convert to true/false to use in confusion matrix
y_pred=(y_pred>.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#make one prediction, here country has to convert to dummy variable to 0,0
#and fit_transform should be applied to one data
new_prediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>.5)


#use k-fold cross validation to measure correct accuracy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
   classifier=Sequential()
   classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
   classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
   classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
   classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
   return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=1)
accuracy=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracy.mean()
variance=accuracy.std()

#tunning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
   classifier=Sequential()
   classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
   classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
   classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
   classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
   return classifier
classifier=KerasClassifier(build_fn=build_classifier)
#creata a dict for hyper-parameter to try with different parameter as below
parameters={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(extimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train,y_train)
#find the best parameter and accuracy from grid_search attribute
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_



accuracy=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracy.mean()
variance=accuracy.std()

  