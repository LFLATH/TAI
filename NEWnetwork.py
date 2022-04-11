#.\env\scripts\activate
import numpy as np
from numpy import genfromtxt
import os

data = np.genfromtxt("data/1722.csv", delimiter=',', skip_header=1)

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

    def backward(self, yerror, rate):
        weights_error = np.dot(yerror, self.input.T)
        self.weights -= rate * weights_error
        self.bias -= rate * yerror
        return np.dot(self.weights.T, yerror)

class activation(layer):

    def __init__(self, activation, dactivation):
        self.act = activation
        self.dact = dactivation

    def forward(self, input):
        self.input = input
        self.output = self.act(self.input)
        return self.output

    def backward(self, yerror, rate):
        #return self.dact(self.input) * yerror
        return np.multiply(yerror, self.dactivation(self.input))



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

