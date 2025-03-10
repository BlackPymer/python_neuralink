import numpy as np
from numpy import ndarray
from Loss.loss import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        """do nothing"""
        super().__init__()

    def _output(self) -> float:
        """
        calculating loss (y-p)^2
        :return: loss
        """
        loss = np.sum(np.pow(self.target - self.prediction, 2))
        return loss

    def _input_grad(self) -> ndarray:
        """
        calculating _output derivative
        :return: gradient
        """
        return 2 * (self.target - self.prediction) / self.prediction.shape[0]
