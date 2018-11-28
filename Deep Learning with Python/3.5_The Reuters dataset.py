# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:32:28 2018

@author: jing.o.li
"""

# In this section, we will focus on predicting more than two classes.
# Since each data point should be classified into only one category, the problem is 
# more specifically an instance of single-label, multiclass classification. 
# If each data point could belong to multiple categories, we'd be facing a multilabel,
# multiclass classification problem.



###############################
##### The Reuters dataset #####
###############################

# Loading the Reuters dataset

from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=1000)

# the argument num_words=1000 restricts the data to the 10,000 most frequently occurring
# words found in the data

max([max(each) for each in train_data])

# in the training set, each example is a list of integers

print(train_data[0])



##############################
##### Preparing the data #####
##############################

# encoding the data

import numpy as np
def vectorize_sequence(sequences, dimension = 1000):
    results = np.zeros((len(sequences), dimension))
    for row, col in enumerate(sequences):
        results[row,col] = 1
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

# to vectorize the labels, there are two possibilities: you can cast the label list as 
# an integer tensor, or you can use one-hot encoding. 

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# note there is a built-in way to do this in Keras

from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)



##############################
##### Building a network #####
##############################

# In a stack of Dense layers like that you’ve been using, each layer can only access 
# information present in the output of the previous layer. If one layer drops some 
# information relevant to the classification problem, this information can never be recovered 
# by later layers: each layer can potentially become an information bottleneck. 
#
# In the previousexample, you used 16-dimensional intermediate layers, but a 16-dimensional
# space maybe too limited to learn to separate 46 different classes: such small layers may 
# act as infor-mation bottlenecks, permanently dropping relevant information.

# model definition

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# there are two other things you should note about ths architecture:
#       1. the network ends with a dense layer of size 46. this means for each input sample,
#          the network will output a 46-dimensional vector. each entry in this vector will encode
#          a different output class.
#       2. the last layer uses a softmax activation. it means that network will output a 
#          probability distribution over the 46 different output classes. for every input sample,
#          the network will produce a 46-dimensional output vector, where output[i] is the 
#          probability that the sample belongs to class i. the 46 scores will sum to 1.
#
# The best loss function to use in this case is categorical_crossentropy. It measures
# the distance between two probability distributions

# compiling the model

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



######################
##### Validation #####
######################

# set apart 1000 samples in the training data to use as a validation set

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# train the network for 20 epochs

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))

# plot the training and validation loss

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the training and validation accuracy

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



###############################################
##### Retraining the network from scratch #####
###############################################

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)



##############################################
##### Generating predictions on new data #####
##############################################

# generating predictions for new data

predictions = model.predict(x_test)

# each entry in predictions is a vector of length 46

predictions[0].shape

# The largest entry is the predicted class—the class with the highest probability

np.argmax(predictions[0])

# the importance of having sufficiently large intermediate layers
# We mentioned earlier that because the final outputs are 46-dimensional, you should
# avoid intermediate layers with many fewer than 46 hidden units.
# Now let’s see whathappens when you introduce an information bottleneck by having 
# intermediate layersthat are significantly less than 46-dimensional: 
# for example, 4-dimensional.
#
# The network now peaks at ~71% validation accuracy, an 8% absolute drop. 
# This dropis mostly due to the fact that you’re trying to compress a lot of 
# information (enoughinformation to recover the separation hyperplanes of 46 classes) 
# into an intermediatespace that is too low-dimensional. 