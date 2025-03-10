import numpy as np
from numpy import ndarray

from Operations.operation import Operation


class ParamOperation(Operation):
    def __init__(self, param: ndarray):
        super().__init__()
        self.param_grad = None
        self.input_grad = None
        self.param_grad = None
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert self.input_grad.shape == self._input.shape
        assert self.param_grad.shape == self.param.shape

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError
