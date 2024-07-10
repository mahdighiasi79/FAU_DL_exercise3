import numpy as np

from . import Base


class TanH(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.activations = None
        self.type = 'TanH'

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        derivative = 1 - np.power(self.activations, 2)
        error_tensor *= derivative
        return error_tensor
