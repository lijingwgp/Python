#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_train(path):
    
    data_gen = ImageDataGenerator(validation_split = 0.25, rescale=1./255)
    train_data = data_gen.flow_from_directory(path, target_size=(150,150), batch_size=16, class_mode='sparse', subset='training', seed=1)
    
    return train_data


def create_model(input_shape):

    model = Sequential()

    model.add(Conv2D(6, (5, 5), padding='same', activation='relu',input_shape=input_shape))
    model.add(AvgPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(AvgPool2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    
    optimizer = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
    
    return model


def train_model(model, train_data, test_data, batch_size=None, epochs=10, steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model


# In[ ]:




