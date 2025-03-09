import numpy as np
from numpy import ndarray


class Operation(object):
    def __init__(self):
        self.input_grad = None
        self.output = None
        self._input = None

    def forward(self, _input: ndarray) -> ndarray:
        self._input = _input
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert self.output.shape == output_grad.shape
        self.input_grad = self._input_grad(output_grad)
        return self.input_grad

    def _output(self):
        raise NotImplementedError

    def _input_grad(self, output_grad: ndarray):
        raise NotImplementedError
