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
    Dense(4,1),
    Hypertan(),
]

'''
x = X[4]
y= Y[4]

l = Dense(4, 4)

l.forwardfeed(x)

'''


train(network, rmse, drmse, X, Y, epochs=100, learning_pace=.0001)#decreasing learning rate gradually decreases error






# attempting to decrease learning rate gradually
'''
i = .001
for i in range(10):
    train(network, mse, dmse, X, Y, epochs=40, learning_pace=i)
    i = i/10
'''

'''

for layer in network:
        o = layer.forwardfeed(x)
        #print(o)

'''
print(predict(network, X[9]))

