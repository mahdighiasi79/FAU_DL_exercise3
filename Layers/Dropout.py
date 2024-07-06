import copy
import numpy as np

from . import Base


class Dropout(Base.BaseLayer):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None
        self.type = 'Dropout'

    def forward(self, input_tensor):
        output_tensor = copy.deepcopy(input_tensor)
        if not self.testing_phase:
            mask = np.random.choice(2, input_tensor.shape, p=[1 - self.probability, self.probability])
            self.mask = mask
            output_tensor *= mask
            output_tensor /= self.probability
        return output_tensor

    def backward(self, error_tensor):
        error_tensor *= self.mask
        error_tensor /= self.probability
        return error_tensor
