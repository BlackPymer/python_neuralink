from typing import List

import numpy as np
from numpy import ndarray

from Layers.layer import Layer
from Loss.loss import Loss


class NeuralNetwork(object):
    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1, learning_rate: float = 0.01):
        """
        Saves Neural Network's parameters
        :param layers: layers of the NN
        :param loss: Loss of the NN
        :param seed: seed of random generating
        :param learning_rate: speed of training neurolink (experimentally 0.01 is the best value)
        """
        self.learning_rate = learning_rate
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in layers:
                setattr(layer, "seed", self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        """
        Moves data forward throw layers
        :param x_batch: input
        :return: output
        """
        x_out = x_batch

        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_grads: ndarray) -> None:
        """
        Moves data backward throw layers
        :param loss_grads: counted loss gradient
        """
        grad = loss_grads

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        """
        forward propagation -> loss counting -> backward propagation
        :param y_batch: expected result
        :param x_batch: input
        :return: counted loss
        """

        predictions = self.forward(x_batch)

        loss = self.loss.forward(prediction=predictions, target=y_batch)

        self.backward(self.loss.backward())

        return loss

    def params(self):
        """
        :return: neuralink parameters
        """
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        """
        :return: grads of the params of the neuralink
        """
        for layer in self.layers:
            yield from layer.param_grads
