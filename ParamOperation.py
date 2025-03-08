import numpy as np
from numpy import ndarray

from Operation import Operation


class ParamOperation(Operation):
    def __init__(self, param: ndarray):
        super().__init__()
        self.param_grad = None
        self._input_grad = None
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        self._input_grad = self._input_grad(output_grad)
        self.param_grad = self.param_grad(output_grad)

        assert self._input_grad.shape == self._input
        assert self.param_grad.shape == self.param

        return self._input_grad

    def param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError
