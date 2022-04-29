#windows: .\env\scripts\activate
import os#Allows us to access pictures and files with Python
from Image_Processing.HSVConverter import convertHue
from Image_Processing.ImageFormatter import *#Import the method from ImageFormatter
import numpy as np
from Network_Files.dense import Dense
from Network_Files.tan import Hypertan
from Network_Files.loss import mse, dmse
from Network_Files.network import train, predict
from Tdata import *

X = numpy.reshape(imgarray, (81, 784, 1))
Y = numpy.reshape(true, (81,1,1))

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
train(network, mse, dmse, X, Y, epochs=5000, learning_pace=.001)#slowly decreasing this number seems to decrease mse well, maybe also casue treating the imgarray as a new mini batch #nvm
