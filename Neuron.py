import numpy
from Math import *
class Neuron:
    def __init__(self, weights, bias): #Inits the weights and bias of the Neuron
        self.weights = weights #Sets the specific instances of bias and weights
        self.bias = bias
    def feedforward(self, inputs):
        #First we add weight to the inputs through taking the dot product of the weights and the inputs
        #The we adde the bias
        #This is then fed into our sigmoid function
        total = numpy.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


