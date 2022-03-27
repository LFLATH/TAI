import numpy

from HSVConverter import huearray
from Neuron import Neuron

x = huearray

for n in range(len(x)):
    weight1 = numpy.random.rand(784)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the first layer
    biases = 0 

    layer1 = Neuron(weight1, biases)

    x = layer1.feedforward(x)