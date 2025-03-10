from Operations.bias_add import BiasAdd
from Operations.operation import Operation
from Operations.sigmoid import Sigmoid
from Operations.weight_multiply import WeightMultiply
from Layers.layer import Layer
import numpy as np
from numpy import ndarray


class Dense(Layer):
    """
    Dense layer
    """
    def __init__(self, neurons: int,activation: Operation = Sigmoid()):
        """
        :param activation: activation is necessary for function
        """
        super().__init__(neurons)
        self.seed = None
        self.activation = activation

    def _setup_layer(self, _input: ndarray) -> None:
        """
        setup operations for dense layer
        """
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # weights
        self.params.append(np.random.randn(_input.shape[1],self.neurons))

        #bias
        self.params.append(np.random.randn(1,self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]

        return None