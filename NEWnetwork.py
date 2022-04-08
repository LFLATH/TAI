#.\env\scripts\activate
import numpy as np
from numpy import genfromtxt
import os

data = np.genfromtxt("data/1722.csv", delimiter=',', skip_header=1)

class layer:
    
    def __init__(self, inputs, outputs, weights=None, biases=None):
        
        if weights is None:
            self.weights = np.random.rand(inputs, outputs)-.5
        else:
            self.weights = weights

        if biases is None:
            self.biases = np.zeros((1, outputs))
        else:
            self.biases = biases        

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, yerror, rate):
        xerror = np.dot(yerror, self.weights.T)
        werror = np.dot(self.input.T, yerror)
        self.weights -= rate * werror
        self.bias -= rate * yerror
        return xerror

class activation:

    def __init__(self, activation, dactivation):
        self.act = activation
        self.dact = dactivation

    def forward(self, input):
        self.input = input
        self.output = self.act(self.input)
        return self.output

    def backward(self, yerror, rate):
        return self.dact(self.input) * yerror


def lin(x):
    return x

def relu(x):
    return np.maximum(0, x)
def drelu(x):
    return (x > 0) * 1

def mse(true, pred):
    return ((true - pred) ** 2)#.mean()
def dmse(true, pred):
    return -2*(true - pred)



x = data[0]
w = [[1], [1],[1]]

input = layer(3, 3, w)
input.forward(x)
act = activation(relu, drelu)
input.output
act.forward(input.output)


pred = o
true = data[1]

mse(true, pred)


w = w - (dmse(true, pred)*drelu(o)).T
