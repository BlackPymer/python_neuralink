import numpy as np
from numpy import ndarray

from Operations.param_operation import ParamOperation


class BiasAdd(ParamOperation):
    """
    Adding bias operation
    """

    def __init__(self, bias: ndarray):
        """
        :param bias: set bias as Operation param
        """
        super().__init__(param=bias)

    def _output(self) -> ndarray:
        """
        :return: input matrix + bias matrix
        """
        assert self._input.shape == self.param.shape
        return self._input + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        :return: ones of input size
        """
        return np.ones_like(self._input) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        calculating param gradient
        """
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
