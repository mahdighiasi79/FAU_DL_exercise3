import numpy as np
from . import Base
from . import Initializers as Init


class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.zeros((input_size + 1, output_size))
        self.weights[:input_size, :] = Init.He.initialize((input_size, output_size), input_size, output_size)
        self.weights[input_size, :] = Init.Constant().initialize((1, output_size), 1, output_size)
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = np.array([])
        self.type = "FullyConnected"

    def initialize(self, weights_initializer, bias_initializer):
        input_size, output_size = self.weights.shape
        input_size -= 1
        weights = weights_initializer.initialize((input_size, output_size), input_size, output_size)
        biases = bias_initializer.initialize((1, output_size), 1, output_size)
        self.weights = np.concatenate((weights, biases), axis=0)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        weights = self.weights[0: len(self.weights) - 1]
        biases = self.weights[len(self.weights) - 1]
        output_tensor = input_tensor @ weights
        output_tensor += biases
        return output_tensor

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, value):
        self._optimizer = value

    optimizer = property(get_optimizer, set_optimizer)

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, value):
        self._gradient_weights = value

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def backward(self, error_tensor):
        gradient_biases = np.sum(error_tensor, axis=0, keepdims=True)
        gradient_weights = self.input_tensor.transpose() @ error_tensor
        self.gradient_weights = np.concatenate((gradient_weights, gradient_biases), axis=0)

        error_tensor = error_tensor @ self.weights[0: len(self.weights) - 1].transpose()

        if self.optimizer is not None:
            self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return error_tensor
