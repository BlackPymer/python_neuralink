import numpy as np

from Layers.dense import Dense
from Loss.mean_squared_error import MeanSquaredError
from Operations.sigmoid import Sigmoid
from neural_network import NeuralNetwork
if __name__ == '__main__':
    nn = NeuralNetwork(
        layers=[Dense(neurons=13,activation=Sigmoid()),Dense(neurons=10)],
        loss = MeanSquaredError()
    )
    print(nn.forward(np.random.rand(1,13)))