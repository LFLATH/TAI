import numpy

from Math import derivative_sigmoid, sigmoid
from Neuron import Neuron
#Major current problem
#How do we put the data from pictures into the network, so that when we are intitializng the network
#That data is the same as when we train the network.



class Network():
       
    def  __init__(self):
        self.t = 5
        self.x = numpy.zeros((784))
        self.hn = 28 # defines number of Hidden Neurons in the first and second hidden layers
        self.output1 = numpy.zeros((self.hn))#defines output array for the hidden layers
        self.sumHidden1 = numpy.zeros((self.hn))
        self.sumHidden2 = numpy.zeros((self.hn))
        self.sigsumHidden1 = numpy.zeros((self.hn))
        self.sigsumHidden2 = numpy.zeros((self.hn))
        self.sumOutput =  0
        self.sigsumOutput =  numpy.zeros(1)
        self.step = 0
        #Data should be an array of HSV arrays
        #This is what we are feeding into the network
        #Results should be an array with corresponding 1's and 0's
        #This would signify the correct result
        self.epochs = 100 #This is the number of times we go through our pictures
        self.learning_pace = 0.1#This affects how drastically we change our weights and biases
        self.partial_weight_output = numpy.zeros(28)
        self.partial_derv_input_bias = numpy.zeros(28)
        self.partial_derv_input = numpy.zeros(21952)
        self.partial_derv_hidden = numpy.zeros(784)
        self.partial_derv_hidden_bias = numpy.zeros(28)
        self.partial_derv_output = numpy.zeros(1)
        self.partial_derv_output_bias = 0
        self.weightsHidden1 = numpy.zeros(784)
        self.weightsHidden2 = numpy.zeros(28)
        self.biasesHidden1 = numpy.zeros(28)
        self.biasesOutput = 0
        self.weightsOutput = numpy.zeros(28)
        self.biasesHidden2 = numpy.zeros(28)
        step = 0 #resets step
        for i in range(self.hn):# applies feedforward method from neuron class to each pixel once for each neuron in the hidden layer(numbers sent from the 784 pixels to the hidden neurons)

            weightsHidden1 = numpy.random.rand(784)-.5 #sets a random weight between -.5 and .5 for each neuron/pixel in the input layer
            biasesHidden1 = 0#sets bias for each neuron        
            layer1 = Neuron(weightsHidden1, biasesHidden1)#defines layer1 as class neuron with the weights and biases defined previously
            result = layer1.feedforward(self.x)#dot product of each pixel in the input and the random weights plus the bias
            self.output1[step] = result#result added to output1 array(copied from your hsv method)
            step = step + 1
        step = 0 #resets step

        for i in range(self.hn):

                weightsHidden2 = numpy.random.rand(28,)-.5 
                biasesHidden2 = 0
                layer2 = Neuron(weightsHidden2, biasesHidden2)
                result = layer2.feedforward(self.output1)
                self.output1[step] = result
                step = step + 1

        fn = 1
        output = numpy.zeros((fn))
        step = 0

        for i in range(fn):

            weightsOutput = numpy.random.rand(1,)-.5 
            biasesOutput = 0
            layer2 = Neuron(weightsOutput, biasesOutput)
            result = layer2.feedforward(output)
            output[step] = result
            step = step + 1

    def train(self, data, results):
        print(self.t)
        for epoch in range(self.epochs):#Loops throught the number of epochs we have
            for array, img_true in zip(data, results):
                #For every array and corresponding correct result we execute this loop
                #The zip function combines the data and the result into a pair
                #Loops through all th connections to the first hidden layer from the input neurons
                #Takes the sum of these connections times the weight
                #Takes the sigmoid of these sums

                for i in range(0, 28):
                    for j in range(0, 784):
                        self.sumHidden1[i] += self.weightsHidden1[j] * array[j]
                    self.sumHidden1[i] += self.biasesHidden1[i]

                for g in  range(0, 28):
                    self.sigsumHidden1[g] = sigmoid(self.sumHidden1[g])
                #Same as previous except for 2nd hidden layer
                for i in range(0, 28):
                    for j in range(0, 28):
                        self.sumHidden2[i] += self.weightsHidden2[j] * self.sumHidden1[j]
                    self.sumHidden2[i] += self.biasesHidden2[i]
                for g in  range(0, 28):
                    self.sigsumHidden2[g] = sigmoid(self.sumHidden2[g])
                #Same as above except for the output layer
                for i in range(0, 28):
                    self.sumOutput += self.weightsOutput[i] * self.sumHidden2
                    self.sumOutput += self.biasesOutput

                sigsumOutput = sigmoid(self.sumOutput)
                #Defining y_pred
                y_pred = sigsumOutput
                #Calculating partial derivative
                #partial_y_pred = -2 * (results - y_pred)
                step = 0
                for i in range(0, 28):
                    for j in range(0, 784):
                        self.partial_derv_input[j + step] = array[j] * derivative_sigmoid(self.sumHidden1[i])
                    step += 784
                    self.partial_derv_input_bias[i] = derivative_sigmoid(self.sumHidden1[i])
                step = 0
                for i in range(0,28):
                    for j in range(0,28):
                        self.partial_derv_hidden[j + step] = self.sumHidden1[i] * derivative_sigmoid(self.sumHidden2[i])
                    step += 28
                    self.partial_derv_hidden_bias[i] = derivative_sigmoid(self.sumHidden2[i])
                for i in range(0,28):
                    self.partial_derv_output = self.sumHidden2[i] * derivative_sigmoid(self.sumOutput)
                self.partial_derv_output_bias = derivative_sigmoid(self.sumOutput)


                



