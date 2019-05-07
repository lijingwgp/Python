# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:59:57 2019

@author: jing.o.li
"""

############################
##### Data Preparation #####
############################

import numpy as np
import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\train"
valid_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\valid"
test_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\test"

# instantiating the VGG16 convnet model
# note that the input_shape is purely optional, if we don't pass it, the network
# will be able to process inputs of any size.

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
conv_base.summary()



##############################################################
##### Extracting features using the pretrained conv base #####
##############################################################

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    
    datagen = ImageDataGenerator(rescale=1/.255)
    generator = datagen.flow_from_directory(
            directory,
            target_size = (150,150),
            batch_size = 20,
            class_mode = 'binary')
    
    i=0
    for inputs_batch, labels_batch in generator:
        feature_batch = conv_base.predict(inputs_batch)
        features[i * 20: (i + 1) * 20] = feature_batch
        labels[i * 20: (i +1) * 20] = labels_batch
        i += 1
        if i * 20 >= sample_count:
            break
    return features, labels

# extract the features
train_features, train_labels = extract_features(train_dir, 2000)
valid_features, validation_labels = extract_features(valid_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# re-shape the features so that can be passed into a classifier
train_features = np.reshape(train_features, (2000,4*4*512))
valid_features = np.reshape(valid_features, (1000,4*4*512))
test_features = np.reshape(test_features, (1000,4*4*512))



##################################################################
##### Defining the training the densely connected classifier #####
##################################################################

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4* 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,epochs=30,batch_size=20,
                    validation_data=(valid_features, validation_labels))

# plot the result
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

