# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 10:55:39 2018

@author: jing.o.li
"""

import theano
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# train_images and train_labels form the training set, the data that the model will learn from. 
# The model will then be tested on the test set which has test_images and test_labels.

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# the images are encoded as numpy arrays, and the labels are an array of digits, ranging from 0 to 9.
# the images and labels have one-to-one correspondence. 

train_images.shape
len(train_labels)
train_images
train_labels

# The workflow will be as follows: First, we’ll feed the neural network the training data,
# train_images and train_labels. The network will then learn to associate images andlabels. 
# Finally, we’ll ask the network to produce predictions for test_images, and we’llverify whether 
# these predictions match the labels from test_labels.

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28,)))
network.add(layers.Dense(10, activation = 'softmax'))

# The core building block of neural networks is the layer, a data-processing module that
# you can think of as a filter for data. Some data goes in, and it comes out in a more useful form.
# 
# Most  ofdeep learning consists of chaining together simple layers that will implement a 
# form of progressive data distillation. A deep-learning model is like a sieve for data 
# processing, made of a succession of increasingly refined data filters—the layers.
#
# Here,  our  network  consists  of  a  sequence  of  two  Dense  layers,  which  are  
# densely connected (also  called  fully  connected) neural layers. 
#
# The  second  (and  last)  layer  is  a10-way softmax layer, which means it will return an array
# of 10 probability scores (summing to 1). Each score will be the probability that the current 
# digit image belongs to one of our 10 digit classes.


# To make the network ready for training, we need to pick three more things, as partof the compilation step:
#
#   A loss function -- how the network will be able to measure its performance on the training data
#   and thus how it will be able to steer itself in the right direction.
#
#   An optimizer -- the mechanism through which the network will update itself based on the data it sees and its loss function.
#
#   Metrics to monitor during training and testing -- here we only care about accuracy

network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])


# Before training, we’ll preprocess the data by reshaping it into the shape the network expects 
# and scaling it so that all values are in the [0,1] interval. Previously, our training images were
# stored in an array of shape (60000,28,28) with values in [0,255]. We transform it into an array
# of shape (60000, 28*28) with values between 0 and 1.

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# we also need to categorically encode the labels.

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

# two quantities are displayed during training: the loss of the network over the training data,
# and the accuracy of the network over the training data.
#
# Now  let’scheck that the model performs well on the test set.

network.evaluate(test_images, test_labels)

