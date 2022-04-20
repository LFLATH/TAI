import numpy as np
from layer import Layer
from activation import Activation

class Hypertan(Activation):
    def __init__(self):
        def hypertan(x):
            return np.tanh(x)
        def hypertan_p(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(hypertan, hypertan_p)