# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:12:24 2019

@author: jing.o.li
"""

############################
##### Data Preparation #####
############################

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# directory where we will store our smaller dataset
origin_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_large"
base_dir = r"C:\Users\jing.o.li\Desktop\cats_and_dogs_small"
os.mkdir(base_dir)

# directory for train, validation, test split
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
valid_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# directory for training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# directory for training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# directory for validation cat pictures
valid_cats_dir = os.path.join(valid_dir, 'cats')
os.mkdir(valid_cats_dir)
# directory for validation dog pictures
valid_dogs_dir = os.path.join(valid_dir, 'dogs')
os.mkdir(valid_dogs_dir)
# directory for test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# directory for test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# copies for the first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(origin_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies for the next 500 cat images to valid_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(origin_dir, fname)
    dst = os.path.join(valid_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies for the next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(origin_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# copies for the first 1000 cat images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(origin_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copies for the next 500 cat images to valid_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(origin_dir, fname)
    dst = os.path.join(valid_dogs_dir, fname)
    shutil.copyfile(src, dst)

# copies for the next 500 cat images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(origin_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(valid_cats_dir)))
print('total validation dog images:', len(os.listdir(valid_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))



###############################
##### Convnet Preparation #####
###############################

#train_cats_dir = r"C:\Users\jing.o.li\Desktop\train\cats"
#train_dogs_dir = r"C:\Users\jing.o.li\Desktop\train\dogs"
#valid_cats_dir = r"C:\Users\jing.o.li\Desktop\valid\cats"
#valid_dogs_dir = r"C:\Users\jing.o.li\Desktop\valid\dogs"
#test_cats_dir = r"C:\Users\jing.o.li\Desktop\test\cats"
#test_dogs_dir = r"C:\Users\jing.o.li\Desktop\test\dogs"

# build a convnet
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
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# configure the model for training
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



##############################
##### Data Preprocessing #####
##############################

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# batches are 20 samples, so it will take 100 batches until you seeyour target of 2,000 samples
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
valid_generator = test_datagen.flow_from_directory(valid_dir,
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

# steps_per_epoch: after having drawn 100 batches from the generator, 
# or alternatively, after run for 100 gradient descent steps, the fitting process will go to the next epoch.
# validation_steps behave similarly
history = model.fit_generator(train_generator, 
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=valid_generator,
                              validation_steps=50)



#################################
##### Visualize Performance #####
#################################

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



#############################
##### Data Augmentaiton #####
#############################

# set up data augmentation configuration
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.2,zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

# display some randomly augmented training images
from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i=0 
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break
plt.show()



##############################
##### Defining a Network #####
##############################

# define a new convnet 
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

# training the convet using data-augmentation generations
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
                                   width_shift_range=0.2,height_shift_range=0.2,
                                   shear_range=0.2,zoom_range=0.2,
                                   horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=100,
                              validation_data=validation_generator,
                              validation_steps=50)

# save the model
model.save('cats_and_dogs_small_2.h5')
