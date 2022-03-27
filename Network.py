import numpy
from HSVConverter import huearray
import Neuron

x = huearray
weights = numpy.random.rand(28,28)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the first layer
biases = numpy.zeros((28,28))#creates biases for each of the 784 neurons

layer1 = Neuron(weights, biases)

layer1.feedforward(x)
