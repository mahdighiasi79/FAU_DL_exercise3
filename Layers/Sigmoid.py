import numpy as np

from . import Base


class Sigmoid(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.activations = None
        self.type = "Sigmoid"

    def forward(self, input_tensor):
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        derivative = self.activations * (1 - self.activations)
        error_tensor *= derivative
        return error_tensor
