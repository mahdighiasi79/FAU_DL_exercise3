import copy
import math
import numpy as np

from . import Initializers as Init
from . import Base


class Conv(Base.BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.convolution_size = np.prod(convolution_shape)
        self.type = "Convolution"

        if len(convolution_shape) == 2:
            weights_shape = [num_kernels, convolution_shape[0], convolution_shape[1]]
        else:
            weights_shape = [num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2]]

        self.weights = Init.UniformRandom.initialize(weights_shape, np.prod(self.convolution_shape),
                                                     np.prod(self.convolution_shape[1:] * num_kernels))
        self.bias = Init.Constant().initialize((num_kernels,), 1, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None

        self.convolution_input = None
        self.x_left_padding = None
        self.x_right_padding = None
        self.y_left_padding = None
        self.y_right_padding = None

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, value):
        self._gradient_weights = value

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_gradient_bias(self, value):
        self._gradient_bias = value

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, value):
        self._optimizer = value

    gradient_weights = property(get_gradient_weights, set_gradient_weights)
    gradient_bias = property(get_gradient_bias, set_gradient_bias)
    optimizer = property(get_optimizer, set_optimizer)

    @staticmethod
    def padding(n, s, f):
        a = ((s - 1) * n) + f
        if (a - s) % 2 == 0:
            right_padding = left_padding = (a - s) / 2
        elif s > 1:
            right_padding = left_padding = (a - s + 1) / 2
        else:
            right_padding = a / 2
            left_padding = (a / 2) - 1
        return int(right_padding), int(left_padding)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]
        x_size = input_tensor.shape[2]
        x_filter_size = self.convolution_shape[1]
        if len(self.stride_shape) == 1:
            x_stride = y_stride = self.stride_shape[0]
        else:
            x_stride, y_stride = self.stride_shape

        if len(input_tensor.shape) == 3:
            self.x_right_padding, self.x_left_padding = self.padding(x_size, 1, x_filter_size)
            right_padding = np.zeros((batch_size, num_channels, self.x_right_padding))
            left_padding = np.zeros((batch_size, num_channels, self.x_left_padding))
            convolution_input = np.concatenate((left_padding, input_tensor, right_padding), axis=2)
            n_x = math.floor((x_size + self.x_right_padding + self.x_left_padding - x_filter_size) / x_stride) + 1

            # performing convolution
            output = []
            for i in range(self.num_kernels):
                channel = []

                for j in range(n_x):
                    start_index = j * x_stride
                    end_index = start_index + x_filter_size

                    convolve = convolution_input[:, :, start_index:end_index] * self.weights[i]
                    convolve = np.sum(convolve, axis=(1, 2), keepdims=False)
                    channel.append(convolve)

                channel = np.array(channel)
                channel += self.bias[i]
                output.append(channel)
            output = np.array(output).transpose(2, 1, 0).transpose(0, 2, 1)

        else:
            y_size = input_tensor.shape[3]
            y_filter_size = self.convolution_shape[2]
            self.x_right_padding, self.x_left_padding = self.padding(x_size, 1, x_filter_size)
            self.y_right_padding, self.y_left_padding = self.padding(y_size, 1, y_filter_size)
            right_padding_x = np.zeros((batch_size, num_channels, self.x_right_padding, y_size))
            left_padding_x = np.zeros((batch_size, num_channels, self.x_left_padding, y_size))
            convolution_input = np.concatenate((left_padding_x, input_tensor, right_padding_x), axis=2)
            right_padding_y = np.zeros(
                (batch_size, num_channels, x_size + self.x_right_padding + self.x_left_padding, self.y_right_padding))
            left_padding_y = np.zeros(
                (batch_size, num_channels, x_size + self.x_right_padding + self.x_left_padding, self.y_left_padding))
            convolution_input = np.concatenate((left_padding_y, convolution_input, right_padding_y), axis=3)
            n_x = math.floor((x_size + self.x_right_padding + self.x_left_padding - x_filter_size) / x_stride) + 1
            n_y = math.floor((y_size + self.y_right_padding + self.y_left_padding - y_filter_size) / y_stride) + 1

            # performing convolution
            output = []
            for i in range(self.num_kernels):
                channel = []

                for j in range(n_x):
                    x_out = []
                    x_start_index = j * x_stride
                    x_end_index = x_start_index + x_filter_size

                    for k in range(n_y):
                        y_start_index = k * y_stride
                        y_end_index = y_start_index + y_filter_size

                        convolve = convolution_input[:, :, x_start_index:x_end_index, y_start_index:y_end_index] * \
                                   self.weights[i]
                        convolve = np.sum(convolve, axis=(1, 2, 3), keepdims=False)
                        x_out.append(convolve)

                    x_out = np.array(x_out)
                    channel.append(x_out)
                channel = np.array(channel)
                channel += self.bias[i]
                output.append(channel)
            output = np.array(output).transpose(0, 1, 3, 2).transpose(0, 2, 1, 3).transpose(1, 0, 2, 3)

        self.convolution_input = convolution_input
        return output

    def backward(self, error_tensor):
        new_error_tensor = np.zeros(self.convolution_input.shape)

        if len(error_tensor.shape) == 3:

            batch_size, num_channels, x_out = error_tensor.shape
            gradient_weights = np.zeros(
                (batch_size, self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2), keepdims=False)

            for i in range(self.num_kernels):
                for j in range(x_out):
                    start_index = j * self.stride_shape[0]
                    end_index = start_index + self.convolution_shape[1]

                    for k in range(start_index, end_index):
                        gradients = copy.deepcopy(self.convolution_input[:, :, k]).transpose() * error_tensor[:, i, j]
                        gradient_weights[:, i, :, k - start_index] += gradients.transpose()

                    for k in range(batch_size):
                        new_error_tensor[k, :, start_index:end_index] += self.weights[i, :, :] * error_tensor[k][i][j]

            self.gradient_weights = np.sum(gradient_weights, axis=0, keepdims=False)
            error_tensor = new_error_tensor[:, :,
                           self.x_right_padding:(self.convolution_input.shape[2] - self.x_left_padding)]

        else:
            if len(self.stride_shape) == 1:
                x_stride = y_stride = self.stride_shape[0]
            else:
                x_stride, y_stride = self.stride_shape
            batch_size, num_channels, x_size, y_size = error_tensor.shape
            gradient_weights = np.zeros((batch_size, self.num_kernels, self.convolution_shape[0],
                                         self.convolution_shape[1], self.convolution_shape[2]))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3), keepdims=False)

            for i in range(num_channels):
                for j in range(x_size):
                    x_start_index = j * x_stride
                    x_end_index = x_start_index + self.convolution_shape[1]

                    for k in range(y_size):
                        y_start_index = k * y_stride
                        y_end_index = y_start_index + self.convolution_shape[2]

                        for l in range(x_start_index, x_end_index):
                            for m in range(y_start_index, y_end_index):
                                gradients = copy.deepcopy(
                                    self.convolution_input[:, :, l, m]).transpose() * error_tensor[:, i, j, k]
                                gradient_weights[:, i, :, l - x_start_index, m - y_start_index] += gradients.transpose()

                        for l in range(batch_size):
                            new_error_tensor[l, :, x_start_index:x_end_index,
                            y_start_index:y_end_index] += self.weights[i, :, :, :] * error_tensor[l][i][j][k]

            self.gradient_weights = np.sum(gradient_weights, axis=0, keepdims=False)
            error_tensor = new_error_tensor[:, :,
                           self.x_right_padding:(self.convolution_input.shape[2] - self.x_left_padding),
                           self.y_right_padding:(self.convolution_input.shape[3] - self.y_left_padding)]

        if self.optimizer is not None:
            weights_optimizer = copy.deepcopy(self.optimizer)
            bias_optimizer = copy.deepcopy(self.optimizer)
            weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            bias_optimizer.calculate_update(self.bias, self.gradient_bias)
        return error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)
