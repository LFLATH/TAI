import numpy
from HSVConverter import makeArray
import Neuron
from PIL import Image



img = Image.open('Test_Images/t1.jpg') #Loads the image
x = makeArray(img)


class Network:

    def __init__(self):
        weights = []
        biases = []
        for i in range(0,23521):# Creates the 23,520 weights we need and puts them into a list
            weights[self.w.format(i)] = numpy.random.normal()
        for j in range(0, 33):#Creates 32 biases. This is because we have 30 neurons in the hidden layer, and 2 output neurons
            biases[self.b.format(j)] = numpy.random.normal()


    weights = numpy.random.rand(28,28)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the first layer
    biases = numpy.zeros((28,28))#creates biases for each of the 784 neurons

    layer1 = Neuron(weights, biases)

    layer1.feedforward(x)
