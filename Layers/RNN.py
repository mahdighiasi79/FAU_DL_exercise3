import numpy as np

from . import Base
from . import FullyConnected as FC
from . import Sigmoid as S
from . import TanH as T


class RNN(Base.BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros((1, hidden_size))
        self.fc_y = FC.FullyConnected(hidden_size, output_size)
        self.fc_h = FC.FullyConnected(input_size + hidden_size, hidden_size)
        self._memorize = False
        self._optimizer = None
        self.type = 'RNN'

        self.input_tensor = None
        self.output_tensor = None
        self.hidden_states = []

    def get_memorize(self):
        return self._memorize

    def get_weights(self):
        return self.fc_h.weights

    def get_gradient_weights(self):
        return self.fc_h.gradient_weights

    def get_optimizer(self):
        return self._optimizer

    def set_memorize(self, value):
        self._memorize = value

    def set_weights(self, value):
        self.fc_h.weights = value

    def set_gradient_weights(self, value):
        self.fc_h.gradient_weights = value

    def set_optimizer(self, value):
        self._optimizer = value

    memorize = property(get_memorize, set_memorize)
    weights = property(get_weights, set_weights)
    gradient_weights = property(get_gradient_weights, set_gradient_weights)
    optimizer = property(get_optimizer, set_optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_h.initialize(weights_initializer, bias_initializer)
        self.fc_y.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        time_steps, sequence_length = input_tensor.shape
        self.input_tensor = input_tensor
        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))

        output = []
        for i in range(time_steps):
            self.hidden_states.append(self.hidden_state)
            x_t = self.input_tensor[i].reshape((1, self.input_size))
            self.hidden_state = self.fc_h.forward(np.concatenate((x_t, self.hidden_state), axis=1))
            self.hidden_state = T.TanH().forward(self.hidden_state)
            y = self.fc_y.forward(self.hidden_state)
            y = S.Sigmoid().forward(y)
            output.append(y[0])
        self.hidden_states.append(self.hidden_state)
        self.output_tensor = np.array(output)
        return self.output_tensor

    def backward(self, error_tensor):
        time_steps, sequence_length = error_tensor.shape

        gradient_weights_y = []
        gradient_weights_h = []
        new_error_tensor = []
        derivative_hidden_state = np.zeros((1, self.hidden_size))

        for i in reversed(range(time_steps)):
            error = error_tensor[i].reshape((1, sequence_length))
            sigmoid = S.Sigmoid()
            sigmoid.activations = self.output_tensor[i].reshape((1, sequence_length))
            error = sigmoid.backward(error)
            self.fc_y.input_tensor = self.hidden_states[i + 1]
            error = self.fc_y.backward(error)
            gradient_weights_y.append(self.fc_y.gradient_weights)
            error += derivative_hidden_state
            tanh = T.TanH()
            tanh.activations = self.hidden_states[i + 1]
            error = tanh.backward(error)
            x_t = self.input_tensor[i].reshape((1, self.input_size))
            self.fc_h.input_tensor = np.concatenate((x_t, self.hidden_states[i]), axis=1)
            error = self.fc_h.backward(error)
            gradient_weights_h.append(self.fc_h.gradient_weights)
            derivative_hidden_state = error[:, self.input_size:]
            new_error_tensor.append(error[0][:self.input_size])

        self.fc_y.gradient_weights = np.sum(np.array(gradient_weights_y), axis=0)
        self.fc_h.gradient_weights = np.sum(np.array(gradient_weights_h), axis=0)
        if self.optimizer is not None:
            self.fc_y.weights = self.optimizer.calculate_update(self.fc_y.weights, self.fc_y.gradient_weights)
            self.fc_h.weights = self.optimizer.calculate_update(self.fc_h.weights, self.fc_h.gradient_weights)

        new_error_tensor = np.flip(np.array(new_error_tensor), axis=0)
        return new_error_tensor
