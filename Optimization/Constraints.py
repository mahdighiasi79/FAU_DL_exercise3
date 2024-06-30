import numpy as np


class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        gradient = np.sign(weights) * self.alpha
        return gradient

    def norm(self, weights):
        w = np.abs(weights)
        w = np.sum(w)
        w *= self.alpha
        return w


class L2_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        gradient = weights * self.alpha
        return gradient

    def norm(self, weights):
        w = np.power(weights, 2)
        w = np.sum(w)
        w *= self.alpha
        return w
