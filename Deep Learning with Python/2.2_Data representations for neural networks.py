# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:00:39 2018

@author: jing.o.li
"""

# Scalars (0D tensors)
# A tensor that contains only one number is called a scalar or 0D tensor.
# 
# You can display the number of axes of a Numpy tensor via the ndim attribute; a scalar tensor 
# has 0 axes (ndim == 0). The number of axes of a tensor is also called its rank.
# Here’s a Numpy scalar

import numpy as np
x = np.array(12)
x
x.ndim

# Vectors (1D tensors)
# An array of numbers is called a vector, or 1D tensor. A 1D tensor is said to have exactlyone axis.

x = np.array([13,2,3,6,9,11])
x
x.ndim

# Matrices (2D tensors)
# An array of vectors is a matrix, or 2D tensor. A matrix has two axes often referred to rows and columns.

x = np.array([[5,4,3],
             [1,2,3],
             [2,2,2]])
x.ndim

# 3D tensors and higher-dimensional tensors
# If you pack such matrices in a new array, you obtain a 3D tensor, which you can visually
# interpret as a cube of numbers.

x = np.array([[[1,2,3],
               [4,3,2]],
              [[2,2,2],
               [1,1,1]]])
x.ndim

# Now let's take a look at the tensor type
# we would display the number of axes of the tensor train_images, the ndim attribute
 
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.ndim
train_images.shape
train_images.dtype

# So what we have here is a 3D tensor of 8-bit integers. More precisely, it’s an array of
# 60,000 matrices of 28 × 8 integers. Each such matrix is a grayscale image, with coefficients between 0
# and 255.

digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)

# Manipulating tensors in Numpy
# In the previous example, we selected a specific digit alongside the first axis using the 
# syntax train_images[i].
# 
# Selecting specific elements in a tensor is called tensor slicing.

my_slice = train_images[10:100]
my_slice.shape

# It’s equivalent to this more detailed notation, which specifies a start index and stop index for the slice along each tensor axis.

my_slice = train_images[10:100, :, :]
my_slice = train_images[10:100, 0:28, 0:28]

# The notion of data batches
# In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll
# come across in deep learning will be the samples axis (sometimes called the samples dimension). 
# In the MNIST example, samples are images of digits.
#
# deep-learning models don’t process an entire dataset at once; rather,they break the data 
# into small batches. Concretely, here’s one batch of our MNIST dig-its, with batch size of 128:

batch1 = train_images[:128]
batch2 = train_images[129:256]

# When considering such a batch tensor, the first axis (axis 0) is called the batch axis 
# or batch dimension. 

# Real-world examples of data tensors
# let's make data tensors more concrete with a few examples similar to what we will encounter later
#
#   vector data: 2D tensors (samples, features)
#   timeseries data: 3D tensors (samples, timesteps, features)
#   images: 4D tensors (samples, height, width, channels)
#   video: 5D tensors (samples, frames, height, width, channels)
#
# See notes for more details

