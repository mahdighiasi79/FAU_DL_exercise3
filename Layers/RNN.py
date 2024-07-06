import numpy as np
import copy

from . import Base
from . import FullyConnected as FC
from . import Sigmoid as S
from . import TanH as T
from . import Initializers as Init


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
        self.k1 = None
        self.k2 = None
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
            self.hidden_state = self.fc_h.forward(np.concatenate((x_t, self.hidden_state), 1))
            self.hidden_state = T.TanH().forward(self.hidden_state)
            y = self.fc_y.forward(self.hidden_state)
            y = S.Sigmoid().forward(y)
            output.append(y[0])
        self.hidden_states.append(self.hidden_state)
        self.output_tensor = np.array(output)
        return self.output_tensor

    def backward(self, error_tensor):
        pass
