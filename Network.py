import numpy

from Math import sigmoid
from Neuron import Neuron
#Major current problem
#How do we put the data from pictures into the network, so that when we are intitializng the network
#That data is the same as when we train the network.

try:
   x
except NameError:
   x = numpy.zeros((784))
    

hn = 28 # defines number of Hidden Neurons in the first and second hidden layers
output1 = numpy.zeros((hn))#defines output array for the hidden layers
sumInput = numpy.zeros((hn))
sumHidden = numpy.zeros((hn))
sumHidden2 =  numpy.zeros(2)
step = 0

for i in range(hn):# applies feedforward method from neuron class to each pixel once for each neuron in the hidden layer(numbers sent from the 784 pixels to the hidden neurons)

    weightsInput = numpy.random.rand(784)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the input layer
    biasesInput = 0#sets bias for each neuron        
    layer1 = Neuron(weightsInput, biasesInput)#defines layer1 as class neuron with the weights and biases defined previously
    result = layer1.feedforward(x)#dot product of each pixel in the input and the random weights plus the bias
    output1[step] = result#result added to output1 array(copied from your hsv method)
    step = step + 1
step = 0 #resets step

for i in range(hn):

        weightsHidden = numpy.random.rand(28,)-.5 
        biasesHidden = 0
        layer2 = Neuron(weightsHidden, biasesHidden)
        result = layer2.feedforward(output1)
        output1[step] = result
        step = step + 1

fn = 1
output = numpy.zeros((fn))
step = 0

for i in range(fn):

    weightsHidden2 = numpy.random.rand(1,)-.5 
    biasesHidden2 = 0
    layer2 = Neuron(weightsHidden2, biasesHidden2)
    result = layer2.feedforward(output)
    output[step] = result
    step = step + 1

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
            #Loops through all th connections to the first hidden layer from the input neurons
            #Takes the sum of these connections times the weight
            #Takes the sigmoid of these sums

            for i in range(0, 29):
                for j in range(0, 785):
                    sumInput[i] += weightsInput[j] * array[j]
                sumInput[i] += biasesInput[i]

            for g in  range(0, 29):
                sumInput[g] = sigmoid(sumInput[g])
            #Same as previous except for 2nd hidden layer
            for i in range(0, 29):
                for j in range(0, 29):
                    sumHidden[i] += weightsHidden[j] * array[j]
                sumHidden[i] += biasesHidden[i]
            for g in  range(0, 29):
                sumHidden[g] = sigmoid(sumHidden[g])
            #Same as above except for the output layer
            for i in range(0, 29):
                for j in range(0, 3):
                    sumHidden2[i] += weightsHidden2[j] * array[j]
                sumHidden2[i] += biasesHidden2[i]

            sumHidden2 = sigmoid(sumHidden2)
            return(sumHidden2)
            #Defining y_pred
            #y_pred = sumHidden2
            #Calculating partial derivative
        # partial_y_pred = -2 * (results - y_pred)



        



