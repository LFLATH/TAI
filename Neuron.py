import numpy

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

def sigmoid(t):
    #We are using a sigmoid activation function
    #This function will compute the final value of our NN
    return 1 / (1 + numpy.exp(-t))

from HSVConverter import huearray

x = huearray

weights = numpy.random.rand(28,28)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the first layer
biases = numpy.zeros((28,28))#creates biases for each of the 784 neurons

layer1 = Neuron(weights, biases)

layer1.feedforward(x)

