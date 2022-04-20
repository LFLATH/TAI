import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)
    def forwardfeed(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    def backwardprop(self, output_grad, learning_pace):
        weights_grad = np.dot(output_grad, self.input.T)
        input_grad = np.dot(self.weights.T, output_grad)
        self.weights -= learning_pace * weights_grad
        self.bias -= learning_pace * output_grad
        return input_grad
