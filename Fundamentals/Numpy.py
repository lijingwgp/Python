# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:05:22 2017

@author: Jing
"""

# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]
# Import the numpy package as np
import numpy as np
# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)


# height is available as a regular list
# Create a numpy array from height: np_height
np_height = np.array(height)
# Convert np_height to m: np_height_m
np_height_m = np_height * 0.0254


# height and weight are available as a regular lists
# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * 0.0254
# Create array from weight with correct units: np_weight_kg
np_weight_kg = np.array(weight) * 0.453592
# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m ** 2


# height and weight are available as a regular lists
# Calculate the BMI: bmi
np_height_m = np.array(height) * 0.0254
np_weight_kg = np.array(weight) * 0.453592
bmi = np_weight_kg / np_height_m ** 2
# Create the light array
light = bmi < 21
# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])


# height and weight are available as a regular lists
# Store weight and height lists as numpy arrays
np_weight = np.array(weight)
np_height = np.array(height)
# Print out the weight at index 50
print(np_weight[50])
# Print out sub-array of np_height: index 100 up to and including index 110
print(np_height[100:110])


# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)
# Print out the shape of np_baseball
print(np_baseball.shape)


# baseball is available as a regular list of lists
# Create np_baseball (2 cols)
np_baseball = np.array(baseball)
# Print out the 50th row of np_baseball
print(np_baseball[49])
# Select the entire second column of np_baseball: np_weight
np_weight = np_baseball[:,1]
# Print out height of 124th player
print(np_baseball[123,0])


# baseball is available as a regular list of lists
# updated is available as 2D numpy array
# Create np_baseball (3 cols)
np_baseball = np.array(baseball)
# Print out addition of np_baseball and updated
print(np_baseball + updated)
# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1])
# Print out product of np_baseball and conversion
print(np_baseball * conversion)


# Print out the mean of np_height
print(np.mean(np_height))
# Print out the median of np_height
print(np.median(np_height))
# Print out the standard deviation on height
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))
# Print out correlation between first and second column
corr = np.corrcoef(np_baseball[:,0], np_baseball[:,1])
print("Correlation: " + str(corr))


# heights and positions are available as lists
# Convert positions and heights to numpy arrays: np_positions, np_heights
np_positions = np.array(positions)
np_heights = np.array(heights)
# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == "GK"]
# Heights of the other players: other_heights
other_heigts = np_heights[np_positions != 'GK']
# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))
# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heigts)))