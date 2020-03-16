# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:08:28 2019

@author: jing.o.li
"""

###################################
##### Previous example -- 5.2 #####
###################################

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\train"
valid_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\valid"
test_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\test"

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
                                   width_shift_range=0.2,height_shift_range=0.2,
                                   shear_range=0.2,zoom_range=0.2,
                                   horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(valid_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')
history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)
model.save('cats_and_dogs_small_2.h5')



################################################
##### Visualizing intermediate activations #####
################################################

from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

# preprocessing a single image
img_path = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small\test\cats\1700.jpg"

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)

# displaying a test picture
plt.imshow(img_tensor[0])
plt.show()

# instantiating a model from an input tensor and a list of output tensors
from keras import models

# extract the outputs of the top eight layers
layer_outputs = [layer.output for layer in model.layer[:8]]

# creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# running the model in predict mode
activations = activation_model.predict(img_tensor)

# this is the activation of the first convolution layer for the cat image input
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# visualizing the fourth and seventh channel
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,4], cmap='viridis')
plt.matshow(first_layer_activation[0,:,:,7], cmap='viridis')