import numpy as np
from Network_Files.layer import Layer

class Activation(Layer):
    def __init__(self, acti, acti_p):
        self.acti = acti
        self.acti_p = acti_p
    def forwardfeed(self, input):
        self.input = input
        return self.acti(self.input)
    def backwardprop(self, output_grad, learning_pace):
        return np.multiply(output_grad, self.acti_p(self.input))
        

