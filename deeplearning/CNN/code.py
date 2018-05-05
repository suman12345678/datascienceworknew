# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:51:35 2018

@author: suman
"""

# Convolutional Neural Network to find cats or dogs
#as keras will take care of of the folder structure like
#         dataset
#           |
#  training   test    
#   |          |
#cats  dogs   cats  dogs
#images imges imges imges 
#so no data preprocessing required
  
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential  #for initializing ann
from keras.layers import Conv2D  #convolution creation
from keras.layers import MaxPooling2D #pooling
from keras.layers import Flatten  #pooling to convert large 1 col vector
from keras.layers import Dense  #add fully connected layers

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution creation 32 is filter/feature maps, 
#3 no of rows 3 no of col in feature detector
#input shape 3 means color pic
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling: reducing size of each feature map by taking max value with 2*2 matrix
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#improve model by adding 2nd convolution layer and pooling 
#but here its not on input image so input_shape not required
#classifier.add(Conv2D(32, (3, 3),  activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Adding a second convolutional layer
#classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection 128 hidden layer in ANN
classifier.add(Dense(units = 128, activation = 'relu'))
# 1 output layer, as output binary in ANN
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN, adam is used to choose stotastic gradint algorithm
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images preprocessing code from keras.io

from keras.preprocessing.image import ImageDataGenerator
#rescale pixel (max 255) between 0-1

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
import os
os.chdir('C:\\Users\\suman\\Desktop\\deep learning\\Convolutional_Neural_Networks')
#mentione the training dataset directory
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),#from input_shape
                                                 batch_size = 32,
                                                 class_mode = 'binary')#cats and dogs so binary

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#fit model to training set careful it takes huge time
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,#total 8000 images to process 1 epoch
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000) #or nb_val_samples=2000

#part 3 making new prediction
import numpy as np
from keras.preprocessing import image
#import image with size same as training i.e. 64,64
test_image=image.load_img('dataset/single_prediction/cat_or_dog_3.jpg',target_size = (64, 64))   
#convert  image to 3 dim from 2 dim i.e. from (64 64) to (64 64 3)                   
test_image=image.img_to_array(test_image)                         
#classifier.predict(test_image) #this will give error 4 dim expected enter one more dimension as batch
test_image=np.expand_dims(test_image,axis=0)#this will convert to 1,64,64,3 dim
result=classifier.predict(test_image) # this will give 1 or 0
#how to know 1 is dog or cat
training_set.class_indices #show 1 is dog or cat
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'

print(prediction)

                         
                         
#save model in working directory
classifier_json=classifier.to_json()
with open("classifier_json","w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("classifier.h5")    
print("saved")
    
#load json
from keras.models import model_from_json
json_file=open('classifier_json','r')
loaded_classifier_json=json_file.read()
json_file.close()
loaded_classifier=model_from_json(loaded_classifier_json)
loaded_classifier.load_weights("classifier.h5")
print("loaded")    