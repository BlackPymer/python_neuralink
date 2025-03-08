import numpy as np
from numpy import ndarray


class Operation(object):
    def __init__(self):
        self.output = None
        self._input = None

    def forward(self, _input: ndarray) -> ndarray:
        self._input = _input
        self.output = self.output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert self.output.shape == output_grad.shape
        self._input_grad = self._input_grad(output_grad)
        return self._input_grad

    def output(self):
        raise NotImplementedError

    def _input_grad(self, output_grad: ndarray):
        raise NotImplementedError
