import numpy as np
import copy

from . import Base


class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.output_tensor = np.array([])
        self.type = "SoftMax"

    def forward(self, input_tensor):
        exp = np.exp(input_tensor - np.max(input_tensor))
        s = np.sum(exp, axis=1, keepdims=True)
        output_tensor = exp / s
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        batch_size, output_size = self.output_tensor.shape
        derivative_matrix = np.repeat(self.output_tensor[:, :, np.newaxis], output_size, axis=2)
        transpose_matrix = copy.deepcopy(derivative_matrix.transpose((0, 2, 1)))
        transpose_matrix *= -derivative_matrix
        identity_matrix = np.identity(output_size)
        identity_matrix = np.repeat(identity_matrix[np.newaxis, :, :], batch_size, axis=0)
        derivative_matrix = transpose_matrix + (identity_matrix * derivative_matrix)
        error_tensor = np.repeat(error_tensor[:, :, np.newaxis], output_size, axis=2)
        error_tensor *= derivative_matrix
        error_tensor = np.sum(error_tensor, axis=1, keepdims=False)
        return error_tensor
