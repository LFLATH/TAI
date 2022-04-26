import os#Allows us to access pictures and files with Python
from HSVConverter import convertHue
from ImageFormatter import *#Import the method from ImageFormatter
import numpy as np
from numpy import genfromtxt
from dense import Dense
from tan import Hypertan
from mse import mse, mse_p, rmse, drmse
from network import train, predict
import tensorflow as tf
import tensorflow_datasets as tfds

data = tfds.load('mnist', split='train', shuffle_files=True)

'''
X = np.genfromtxt("C:/Users/Sweet/source/repos/Weather/data/1722.csv", int, delimiter=',', skip_header=1)
Y = np.genfromtxt("C:/Users/Sweet/source/repos/Weather/data/1722.csv", int, delimiter=',', skip_header=2)
X = X[0:100]
Y = Y[0:100]
'''

network = [
    Dense(3, 3),
    Hypertan(),
    Dense(3, 3),
    Hypertan(),
]


train(network, rmse, drmse, X, Y, epochs=100000, learning_pace=.001)#slowly decreasing this number seems to decrease mse well, maybe casue it is treating the imgarray as a new mini batch #nvm


'''
epochs=10
learning_pace=.001

for epoch in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        for x_sub in x:
            output = predict(network, x_sub)
            
            error += mse(y, output)
            
            grad = mse_p(y, output)
            for layer in reversed(network):
                grad = layer.backwardprop(grad, learning_pace)
    print(f"{epoch + 1}/{epochs}, error={error}")

'''
