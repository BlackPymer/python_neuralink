from pathlib import Path

import numpy as np

from Layers.dense import Dense
from Loss.mean_squared_error import MeanSquaredError
from Operations.sigmoid import Sigmoid
from Optimizers.stochastic_gradient_descent import SGD
from Trainer import Trainer
from common import *

reset = False # start learning from the very beginning, deleting previous data

# parameters of NN
batch_size = 1024  # a number of cases to train neural network with
learning_rate = 0.01 # speed of learning (0.01 is experimentally recommended)
test_cases = 16 # a number of cases to test neural network with
max_epochs = 5120 # max number of epochs to train
eval_every = 20 # every "eval_every" the progress of learning is checked





if __name__ == '__main__':
    file_exists = Path(data_file_name).exists()
    # creating training-cases
    x_train = np.random.random((batch_size, 4))
    y_train = np.array([np.array(cmyk_to_rgb(*x))/255 for x in x_train])

    # creating test-cases
    x_test = np.random.random((test_cases, 4))
    y_test = np.array([np.array(cmyk_to_rgb(*x))/255 for x in x_test])

    # Initialize or load the neural network
    if file_exists and not reset:
        neural_network = deepcopy(load_from_file(data_file_name))
    else:
        neural_network = NeuralNetwork(
            layers=[Dense(neurons=13, activation=Sigmoid()),
                    Dense(neurons=3)],
            loss=MeanSquaredError(),
            learning_rate=learning_rate
        )
    optimizer = SGD(learning_rate=learning_rate)
    trainer = Trainer(neural_network, optimizer)

    # Fit the model
    trainer.fit(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        epochs=max_epochs,
        eval_every=eval_every,
        restart=reset
    )

    # Save the trained network
    save_to_file(neural_network, data_file_name)





