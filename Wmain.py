import os#Allows us to access pictures and files with Python
import numpy as np
from dense import Dense
from tan import Hypertan
from loss import mse, dmse, rmse, drmse
from network import train, predict
from data import data, true

X = np.reshape(data, (150, 4, 1))
Y = np.reshape(true, (150, 1, 1))

network = [
    Dense(4, 4),
    Hypertan(),
    Dense(4,4),
    Hypertan(),
    Dense(4, 4),
    Hypertan(),
    Dense(4, 4),
    Hypertan(),
    Dense(4, 4),
    Hypertan(),
    Dense(4, 1),
    Hypertan(),
]


train(network, mse, dmse, X, Y, epochs=10000, learning_pace=.001)

print(predict(network, X[5]))
print(predict(network, X[105]))




# attempting to decrease learning rate gradually
'''
i = .01
for i in range(100):
    i = i/5
    for j in range(100):
        train(network, mse, dmse, X, Y, epochs=100, learning_pace=i)
        i = i*2
'''

