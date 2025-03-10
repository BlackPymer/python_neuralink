from numpy import ndarray
import numpy as np
from Operations.operation import Operation


class Sigmoid(Operation):
    """sigmoid operation in Neural Network"""

    def _output(self) -> ndarray:
        """calculate sigmoid function"""
        return 1 / (1 + np.exp(-self._input))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        """
        sigmoid backward calculates as sigmoid(x) * (1-sigmoid(x))
        """
        return self.output * (1 - self.output) * output_grad
