# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:54:56 2018

@author: jing.o.li
"""

#################################
#### loading the IMD dataset ####
#################################

from keras.datasets import imdb
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words = 10000)

# the argument num_words = 10000 means you'll only keep the top 10000 most frequently occurring
# words in the training data
#
# The variables train_data and test_data are lists of reviews; each review is a list 
# of word indices (encoding  a  sequence  of  words).
#
# train_labels and test_labels are lists of 0s and 1s, where 0 stands for negative 
# and 1 stands for positive
#
# here's how you can quickly decode one of these reviews back to English words:

word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
        [reverse_word_index.get(i-3,'?') for i in train_data[0]])



############################
#### Preparing the data ####
############################

# Because you’re restricting yourself to the top 10,000 most frequent words,
# no word index will exceed 10,000:

max([max(each) for each in train_data])

# you can feed lists of integers into a neural network. you have to turn your lists into tensors. 
#       1. pad your lists so that they all have the same length, turn them into a 
#          integer tensor of shape (sample, word_indices)
#       2. one-hot encode your list to turn them into vectors of 0s and 1s

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for row, col in enumerate(sequences):
        results[row, col] = 1
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



###############################
#### Building your network ####
###############################

from keras import models, layers, optimizers, losses, metrics

# model definition

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# compiling the model

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# sometimes you may want to configure the parameters of your optimizer or pass a 
# custom loss function or metric function. the former can be done by passing an 
# optimizer class instance as the optimizer argument; the latter can bedone by 
# passing function objects as the loss and/or metrics arguments 

# configuring the optimizer and using custom losses/metrics
#model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#              loss=losses.binary_crossentropy,
#              metrics=[metrics.binary_accuracy])



#############################
#### Validating approach ####
#############################

import matplotlib.pyplot as plt

# setting aside a validation set

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# now train the model for 20 iterations over all samples, in mini-batches of 512 samples.
# at the same time, we'll monitor loss and accuracy on the 10,000 samples that we set apart.

# training the model

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))

# accessing training history

history_dict = history.history
history_dict.keys()
[u'acc', u'loss', u'val_acc', u'val_loss']

# note that the call to model.fit() returns a History object. This object has a member 
# history, which is a dictionary containing data about everything that happenedduring 
# training. 

# plotting the training and validation loss

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# plotting the training and validation accuracy

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# as one may observe, the training loss decreases with every epoch, and the training 
# accuracyincreases with every epoch. That’s what you would expect when running 
# gradient-descent optimization—the quantity you’re trying to minimize should be less 
# with every iteration.

# But that isn’t the case for the validation loss and accuracy: they seem to peak at the 
# fourth epoch. This is an example of what we warned against earlier: amodel that performs
# better on the training data isn’t necessarily a model that will do better on data it has
# never seen before. 

# In precise terms, what you’re seeing is overfit-ting: after the second epoch, you’re 
# overoptimizing on the training data, and you endup learning representations that are 
# specific to the training data and don’t generalizeto data outside of the training set.

