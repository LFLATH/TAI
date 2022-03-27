import numpy

from HSVConverter import huearray
from Neuron import Neuron

x = huearray

hn1 = 28 # defines number of Hidden Neurons in the first hidden layer

output1 = numpy.zeros((hn1))#defines 1st output array
step = 0

for i in range(hn1):# applies feedforward method from neuron class to each pixel once for each neuron in the hidden layer(connects the 784 pixels to the number of hidden neurons)

    weights = numpy.random.rand(784)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the input layer
    biases = 0#sets bias for each neuron
    layer1 = Neuron(weights, biases)#defines layer1 as class neuron with the weights and biases defined previously
    result = layer1.feedforward(x)#dot product of each pixel in the input and the random weights plus the bias
    output1[step] = result#result added to output1 array(copied from your hsv method)
    step = step + 1
print(output1)#prints the output of feedforward for each neuron in the hidden layer