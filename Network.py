import numpy

from HSVConverter import huearray
from Neuron import Neuron

x = huearray#input

hn = 28 # defines number of Hidden Neurons in the first and second hidden layers
output1 = numpy.zeros((hn))#defines output array for the hidden layers
sumarray = numpy.zeros((hn))
step = 0

for i in range(hn):# applies feedforward method from neuron class to each pixel once for each neuron in the hidden layer(numbers sent from the 784 pixels to the hidden neurons)

    weightsInput = numpy.random.rand(784)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the input layer
    biases = 0#sets bias for each neuron
    layer1 = Neuron(weightsInput, biases)#defines layer1 as class neuron with the weights and biases defined previously
    result = layer1.feedforward(x)#dot product of each pixel in the input and the random weights plus the bias
    output1[step] = result#result added to output1 array(copied from your hsv method)
    step = step + 1
print(output1)#prints feedforward of each input once for each neuron in the hidden layer

step = 0 #resets step

for i in range(hn):

    weightsHidden = numpy.random.rand(28,)-.5 
    biases = 0
    layer2 = Neuron(weightsHidden, biases)
    result = layer2.feedforward(output1)
    output1[step] = result
    step = step + 1
print(output1)

fn = 2
output = numpy.zeros((fn))
step = 0

for i in range(fn):

    weightsHidden2 = numpy.random.rand(2,)-.5 
    biases = 0
    layer2 = Neuron(weightsHidden2, biases)
    result = layer2.feedforward(output)
    output[step] = result
    step = step + 1
print(output)

def train(self, data, results):
    #Data should be an array of HSV arrays
    #This is what we are feeding into the network
    #Results should be an array with corresponding 1's and 0's
    #This would signify the correct result
    epochs = 100 #This is the number of times we go through our pictures
    learnin_pace = 0.1#This affects how drastically we change our weights and biases
    for epoch in range(epochs):#Loops throught the number of epochs we have
        for array, img_true in zip(data, results):
            #For every array and corresponding correct result we execute this loop
            #The zip function combines the data and the result into a pair
            for i in range(0, 29):
                for j in range(0, 785):
                    sum[i] = 
