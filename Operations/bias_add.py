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
        Returns the result of adding bias to _input.
        If bias is a 1D array, reshapes it to (1, n_units) for broadcasting.
        """
        # Ensure bias is a 2D row vector
        if self.param.ndim == 1:
            bias = self.param.reshape(1, -1)
        else:
            bias = self.param

        # Better assertion: ensure the number of features matches
        assert self._input.shape[1] == bias.shape[1], (
            f"Input has {self._input.shape[1]} features but bias has {bias.shape[1]}"
        )

        return self._input + bias

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
