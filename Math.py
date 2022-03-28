import numpy


def sigmoid(t):
    #We are using a sigmoid activation function
    #This function will compute the final value of our NN
    return 1 / (1 + numpy.exp(-t))
def derivative_sigmoid(t):
    #Derivative of the sigmoid function
    func = sigmoid(t)
    return func * (1-func)
def loss(ripe_true, ripe_pred):
    #Calculate the current loss 
    return ((ripe_true - ripe_pred) ** 2).mean()





