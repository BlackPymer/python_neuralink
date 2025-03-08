import numpy as np
from numpy import ndarray
from ParamOperation import ParamOperation


class WeightMultiply(ParamOperation):
    def __init__(self, weights: ndarray):
        super().__init__(weights)

    def output(self):
        return np.dot(self._input, self.param)

    def _input_grad(self, output_grad: ndarray):
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(np.transpose(self._input, (1, 0)), output_grad)
