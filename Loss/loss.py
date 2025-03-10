import numpy as np
from numpy import ndarray


class Loss(object):
    """
    Loss of NN
    """

    def __init__(self):
        """
        Do nothing
        """
        self.input_grad = None
        self.target = None
        self.prediction = None

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        """
        Calculating loss value
        :param prediction: the output of forward of Neural Network
        :param target: expected result
        :return: loss value
        """
        assert prediction.shape == target.shape

        self.prediction = prediction
        self.target = target

        loss_value = self._output()
        return loss_value

    def backward(self) -> ndarray:
        """
        calculating gradient of input
        :return: gradient
        """
        self.input_grad = self._input_grad()
        assert self.input_grad.shape == self.prediction.shape
        return self.input_grad

    def _output(self) -> float:
        """
        Every loss subclasses must realise its own loss counting
        """
        raise NotImplementedError

    def _input_grad(self) -> ndarray:
        """
        Every loss subclasses must realise its own input gradient counting
        """
        raise NotImplementedError
