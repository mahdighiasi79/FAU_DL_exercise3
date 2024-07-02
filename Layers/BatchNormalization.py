import copy
import numpy as np

from . import Base
from . import Helpers as H


class BatchNormalization(Base.BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.weights = np.ones((self.channels,))
        self.bias = np.zeros((self.channels,))
        self.optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.mu = None
        self.rho = None
        self.alpha = 0.8
        self.epsilon = 1e-10

        self.x_hat = None
        self.input_tensor = None
        self.mean = None
        self.variance = None

    def initialize(self):
        self.weights = np.ones((self.channels,))
        self.bias = np.zeros((self.channels,))

    def reformat(self, tensor):
        if len(tensor.shape) == 2:
            if len(self.input_tensor.shape) == 3:
                batch_size, channels, height = self.input_tensor.shape
                tensor = tensor.reshape((batch_size, height, channels))
                tensor = tensor.transpose(0, 2, 1)
            elif len(self.input_tensor.shape) == 4:
                batch_size, channels, height, width = self.input_tensor.shape
                tensor = tensor.reshape((batch_size, height, width, channels))
                tensor = tensor.transpose(0, 1, 3, 2)
                tensor = tensor.transpose(0, 2, 1, 3)
        elif len(tensor.shape) == 3:
            batch_size, channels, height = tensor.shape
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape((batch_size * height, channels))
        else:
            batch_size, channels, height, width = tensor.shape
            tensor = tensor.transpose(0, 2, 1, 3)
            tensor = tensor.transpose(0, 1, 3, 2)
            tensor = tensor.reshape((batch_size * height * width, channels))
        return tensor

    def forward(self, input_tensor):
        self.input_tensor = copy.deepcopy(input_tensor)
        input_tensor = self.reformat(input_tensor)

        if self.testing_phase:
            x_hat = (input_tensor - self.mu) / np.sqrt(self.rho + self.epsilon)
        else:
            mean = np.sum(input_tensor, axis=0) / len(input_tensor)
            variance = np.sum((input_tensor - mean) ** 2, axis=0) / len(input_tensor)
            x_hat = (input_tensor - mean) / np.sqrt(variance + self.epsilon)

            if self.mu is None:
                self.mu = mean
                self.rho = variance
            else:
                self.mu = (self.alpha * self.mu) + ((1 - self.alpha) * mean)
                self.rho = (self.alpha * self.rho) + ((1 - self.alpha) * variance)

            self.x_hat = x_hat
            self.mean = mean
            self.variance = variance

        y_hat = (x_hat * self.weights) + self.bias
        y_hat = self.reformat(y_hat)
        return y_hat

    def backward(self, error_tensor):
        error_tensor = self.reformat(error_tensor)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        self.gradient_weights = error_tensor * self.x_hat
        self.gradient_weights = np.sum(self.gradient_weights, axis=0)

        if self.optimizer is not None:
            self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.optimizer.calculate_update(self.bias, self.gradient_bias)

        input_tensor = self.reformat(copy.deepcopy(self.input_tensor))
        error_tensor = H.compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean, self.variance)
        error_tensor = self.reformat(error_tensor)
        return error_tensor
