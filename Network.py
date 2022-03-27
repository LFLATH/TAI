import numpy
import Neuron
from HSVConverter import huearray
    
class Network:

  def __init__(self):
    weights = numpy.random.rand(28, 28)
    bias = 0
