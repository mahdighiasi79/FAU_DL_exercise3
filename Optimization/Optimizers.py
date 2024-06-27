import numpy as np


class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor -= self.learning_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum_vector = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.momentum_vector = (self.momentum_rate * self.momentum_vector) - (self.learning_rate * gradient_tensor)
        weight_tensor += self.momentum_vector
        return weight_tensor


class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.mu * self.v) + ((1 - self.mu) * gradient_tensor)
        self.r = (self.rho * self.r) + ((1 - self.rho) * gradient_tensor * gradient_tensor)
        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))
        self.k += 1
        weight_tensor -= self.learning_rate * v_hat / (np.power(r_hat, 0.5) + np.finfo(float).eps)
        return weight_tensor
