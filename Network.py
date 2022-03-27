import numpy
import Neuron
from HSVConverter import huearray
    
class Network:

  def __init__(self):
    weights = numpy.random.rand(28, 28)
    bias = 0

    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(numpy.array([out_h1, out_h2]))

    return out_o1

network = Network()
network.feedforward(huearray)
