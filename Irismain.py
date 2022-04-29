import os#Allows us to access pictures and files with Python
import numpy as np
from Network_Files.dense import Dense
from Network_Files.tan import Hypertan
from Network_Files.loss import mse, dmse, rmse, drmse
from Network_Files.network import train, predict
from Irisdata import data, true
import matplotlib.pyplot as plt


X = np.reshape(data, (150, 4, 1))
X[5].shape
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


train(network, mse, dmse, X, Y, epochs=1000, learning_pace=.001)

print(predict(network, X[5]))
print(predict(network, X[105]))

'''
x = X[4]
y= Y[4]

l = Dense(4, 4)

l.forwardfeed(x)


train(network, mse, dmse, X, Y, epochs=1000, learning_pace=.001)#decreasing learning rate gradually decreases error


guesses = []
guesses.append(predict(network, X[0]))
guesses.append(predict(network, X[100]))

for guess in guesses:
    guesser = {
        "Species 3": abs(1 - guess),
        "Species 2": abs(0.666 - guess),
        "Species 1": abs(0.333- guess)
    }
    print(min(guesser, key=guesser.get))
# attempting to decrease learning rate gradually

i = .01
for i in range(10):
    i = i/5
    for j in range(10):
        train(network, mse, dmse, X, Y, epochs=100, learning_pace=i)
        i = i*2
'''