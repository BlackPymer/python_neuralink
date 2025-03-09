import numpy as np
from numpy import ndarray
from param_operation import ParamOperation


class WeightMultiply(ParamOperation):
    """weights multiplying in Neural Network"""

    def __init__(self, weights: ndarray):
        """
        initializing class with self.param = weights
        """
        super().__init__(weights)

    def _output(self) -> ndarray:
        """
        multiplying _input matrix on weights (params)
        """
        assert self.param.shape[0] == self._input.shape[1]
        return np.dot(self._input, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        calculating output gradient
        """
        assert output_grad.shape[1] == self.param.T.shape[0]
        print(f"input grad: {np.dot(output_grad, self.param.T)}")
        return np.dot(output_grad, self.param.T)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        calculating weights gradient
        """
        return np.dot(self._input.T, output_grad)
