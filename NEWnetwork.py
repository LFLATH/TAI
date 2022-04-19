#.\env\scripts\activate
import numpy as np
from numpy import genfromtxt
import os

data = np.genfromtxt("data/1722.csv", delimiter=',', skip_header=1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)
def drelu(x):
    return (x > 0) * 1

def mse(true, pred):
    return ((true - pred) ** 2).mean()
def dmse(true, pred):
    return 2*(true - pred) / np.size(true)


class layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        pass
    def backward(self, output_gradient, rate):
        pass

class connected(layer):
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)-.5
        self.biases = np.random.randn(output_size, 1)       

    def forward(self, input):
        self.input= input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, yerror, rate):
        weights_error = np.dot(yerror, self.input.T)
        self.weights -= rate * weights_error
        self.biases -= rate * yerror
        return np.dot(self.weights.T, yerror)

class activation(layer):

    def __init__(self):
        pass
    def forward(self, input):
        self.input = input
        self.output = sigmoid(self.input)
        return self.output

    def backward(self, yerror, rate):
        #return self.dact(self.input) * yerror
        return np.multiply(yerror, dsigmoid(self.input))



#input = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
#true = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


input = np.array([2, 4, 6, 8])
true = np.array([4, 16, 36, 64])

network = [
    connected(4, 4),
    activation(),

    connected(4, 1),
    activation()
]

epochs = 100
rate = .1

for i in range(epochs):
    error = 0
    for j in range(len(input)):
        x = input[j]
        for layer in network:
            
            output = layer.forward(input)

            error += mse(true, input)

            gd = dmse(true, input)
            for layer in reversed(network):
                gd = layer.backward(gd, rate)

    error /= len(input)
    print('%d/%d, error=%d' % (i + 1, epochs, error))

gd = dmse(true, input)
layer.backward(gd, rate)