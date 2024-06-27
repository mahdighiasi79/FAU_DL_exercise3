import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.prediction_tensor = np.array([])
        self.type = "CrossEntropyLoss"

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = label_tensor * -np.log(prediction_tensor + np.finfo(float).eps)
        loss = np.sum(loss, axis=0, keepdims=False)
        loss = np.sum(loss, axis=0, keepdims=False)
        return loss

    def backward(self, label_tensor):
        error_tensor = (-label_tensor) / (self.prediction_tensor + np.finfo(float).eps)
        return error_tensor
