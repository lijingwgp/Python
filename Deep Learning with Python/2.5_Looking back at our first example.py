# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:44:01 2018

@author: jing.o.li
"""

# We should have a general understanding on behind the scenes in a neural network.
# Let's go back to the first example and review each piece of it.
#
# This was the input data

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Now you understand that the input images are stored in Numpy tensors,  
# which arehere formatted as float32 tensors of shape (60000,784) (training data) and 
# (10000,784) (test data), respectively.

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# This network consists of a chain of two Dense layers, that each layer applies a few simple 
# tensor operations to the input data, and that theseoperations involve weight tensors. 
# Weight tensors, which are attributes of the layers,are where the knowledge of the 
# network persists.

network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

# Now you understand that categorical_crossentropy is the loss function that’s used as a 
# feedback signal for learning the weight tensors, and which the training phase will attempt 
# to minimize. You also know that this reduction of the loss happens via mini-batch stochastic 
# gradient descent. 
#
# The exact rules governing a specific use of gradient descent are defined by the rmsprop 
# optimizer passed as the first argument.

network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Now you understand what happens when you call fit: the network will start to iterateon the 
# training data in mini-batches of 128 samples, 5 times over 
# (each iteration overall the training data is called an epoch). 
# At each iteration, the network will compute thegradients of the weights with regard to the 
# loss on the batch, and update the weights accordingly. 
#
# After these 5 epochs, the network will have performed 2,345 gradientupdates (469 per epoch), 
# and the loss of the network will be sufficiently low that thenetwork will be capable of 
# classifying handwritten digits with high accuracy.
