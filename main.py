#windows: .\env\scripts\activate
import os#Allows us to access pictures and files with Python
from HSVConverter import convertHue
from ImageFormatter import *#Import the method from ImageFormatter
import numpy as np
from dense import Dense
from tan import Hypertan
from loss import mse, dmse
from network import train, predict

img1 = Image.open('Test_Images/tomato1.jpg') #Loads the image
img1 = convertHue(img1)
img2 = Image.open('Test_Images/unripe.jpg') #Loads the image
img2 = convertHue(img2)


imgarray = numpy.array([img1, img2])
imgarray_reshape = numpy.reshape(imgarray, (2, 784, 1))
true_data = numpy.array([1,0])
X = imgarray_reshape
Y = numpy.reshape()
print(numpy.shape(imgarray[0]))

network = [
    Dense(784, 10),
    Hypertan(),
    Dense(10, 10),
    Hypertan(),
    Dense(10, 10),
    Hypertan(),
    Dense(10, 10),
    Hypertan(),
    Dense(10, 10),
    Hypertan(),
    Dense(10, 10),
    Hypertan(),
    Dense(10, 1),
    Hypertan()
]

# train
train(network, mse, dmse, X, Y, epochs=1, learning_pace=.0001)#slowly decreasing this number seems to decrease mse well, maybe also casue treating the imgarray as a new mini batch #nvm

