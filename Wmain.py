import os#Allows us to access pictures and files with Python
import numpy as np
from dense import Dense
from tan import Hypertan
from loss import mse, dmse, rmse, drmse
from network import train, predict
from data import data, true

X = data
Y = true

network = [
    Dense(3, 3),
    Hypertan(),
    Dense(3, 3),
    Hypertan(),
]


#train(network, rmse, drmse, X, Y, epochs=1000, learning_pace=.0001)#decreasing learning rate gradually decreases error

# attempting to decrease learning rate gradually
i = .001

for i in range(10):
    train(network, rmse, drmse, X, Y, epochs=40, learning_pace=i)
    i = i/10
