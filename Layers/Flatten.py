import numpy as np

from . import Base


class Flatten(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.type = "Flatten"

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape((self.input_shape[0], np.prod(self.input_shape[1:])))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
