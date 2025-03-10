from typing import List
import numpy as np
from Operations.operation import Operation
from numpy import ndarray

from Operations.param_operation import ParamOperation


class Layer(object):
    """
    Layer of neurons of Neural Network
    """

    def __init__(self, neurons: int):
        """
        :param neurons: width of the Layer
        """
        self.input_grad = None
        self.output = None
        self.input = None
        self.neurons = neurons
        self.first = True
        self.operations: List[Operation] = []
        self.params: List[ndarray] = []
        self.param_grads: List[ndarray] = []

    def _setup_layer(self, _input: ndarray) -> None:
        """
        setups params on operations (if necessary)
        :param _input: shape of params
        """
        raise NotImplementedError

    def forward(self, _input: ndarray) -> ndarray:
        """
        calculate input throw forward operations
        :return: calculated output
        """
        if self.first:
            self._setup_layer(_input)
            self.first = False
        self.input = _input

        for operation in self.operations:
            _input = operation.forward(_input)

        self.output = _input
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        calculates gradients for input and parameters of operations
        :return: input gradient
        """
        assert self.output.shape == output_grad.shape
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        self.input_grad = output_grad

        self._param_grads()
        return self.input_grad

    def _param_grads(self) -> None:
        """
        get _param_grads from operations
        """
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> None:
        """
        get _param from operations
        """
        self.param = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
