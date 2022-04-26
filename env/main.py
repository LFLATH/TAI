import os#Allows us to access pictures and files with Python
from HSVConverter import convertHue
from ImageFormatter import *#Import the method from ImageFormatter
import numpy as np
from dense import Dense
from tan import Hypertan
from mse import mse, mse_p
from network import train, predict
'''
img_folder = 'Test_Images'#Defines the folder with the test images
for images in os.listdir(img_folder):#Loops through these images
    img = Image.open("Test_Images/"+images)#Opens the Images
    format_Image(img)#Formats the Images

results= numpy.array([1, 1])
'''
img1 = Image.open('Test_Images/21S151-03.jpg') #Loads the image
img1 = convertHue(img1)
img2 = Image.open('Test_Images/t2.jpg') #Loads the image
img2 = convertHue(img2)


imgarray = numpy.array([img1, img2])
for img in imgarray:
    img = img.T
X = imgarray
Y = np.array([0, 1])

import tensorflow as tf
import tensorflow_datasets as tfds
# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='train', shuffle_files=True)


network = [
    Dense(784, 10), #Turned 1,10 into 784, 10
    Hypertan(),
    Dense(10, 1), #Turend 10 , 1 into 7840, 1
    Hypertan()
]

# train
train(network, mse, mse_p, X, Y, epochs=1, learning_pace=.0001)#slowly decreasing this number seems to decrease mse well, maybe also casue treating the imgarray as a new mini batch #nvm

