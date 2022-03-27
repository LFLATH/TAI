import numpy

from HSVConverter import huearray
from Neuron import Neuron

x = huearray

NN = 28 # number of neurons in the first hidden layer

output = numpy.zeros((1,NN))

for i in range(NN):

    weights = numpy.random.rand(784)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the first layer
    biases = numpy.zeros(784)
    layer1 = Neuron(weights, biases)
    output = layer1.feedforward(x)
    output[i] = output
    
