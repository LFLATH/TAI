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
    def backward(self, output_gradient, learning_rate):
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
        self.bias -= rate * yerror
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



input = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
true = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

true = [0, 1, 1, 0]

pred = [.1,.6,.9, .4]

mse(true, pred)


network = [
    connected(2, 3),
    activation(),

    connected(3, 1),
    activation()
]

epochs = 100
rate = .1

for i in range(epochs):
    error = 0
    for x, y in zip(input, true):
        input = x
        for layer in network:
            
            output = layer.forward(output)

            error += mse(true, output)

            gd = dmse(true, output)
            for layer in reversed(network):
                gd = layer.backward(gd, rate)

    error /= len(input)
    print('%d/%d, error=%d' % (i + 1, epochs, error))