import numpy as np
import pickle
from pathlib import Path
from copy import deepcopy
from Layers.dense import Dense
from Loss.mean_squared_error import MeanSquaredError
from Operations.sigmoid import Sigmoid
from Optimizers.stochastic_gradient_descent import SGD
from Trainer import Trainer
from neural_network import NeuralNetwork

data_file_name = "data.pkl"
reset = False # start learning from the very beginning, deleting previous data
# params
batch_size = 512  # a number of cases to train neural network with
learning_rate = 0.01 # speed of learning (0.01 is experimentally recommended)
test_cases = 16 # a number of cases to test neural network with
max_epochs = 5120 # max number of epoches to train
eval_every = 20 # every "eval_every" the progress of learning is checked


def save_to_file(nn: NeuralNetwork, file_name):
    """
    Save the object to a file using pickle.

    Parameters:
        :param file_name: The name of the file to save the object in.
        :param nn: NeuralNetwork to save
    """
    copied_nn = deepcopy(nn)  # Create a deep copy of the neural network
    with open(file_name, 'wb') as file:
        pickle.dump(copied_nn, file)
    print(f"Data saved to {file_name}")


def load_from_file(file_name):
    """
    Load an object from a file using pickle.

    Parameters:
        file_name (str): The name of the file to load the object from.

    Returns:
        ExampleClass: The loaded object.
    """

    with open(file_name, 'rb') as file:
        loaded_object = pickle.load(file)
    print(f"Data loaded from {file_name}")
    return loaded_object


def cmyk_to_rgb(c, m, y, k):
    """
    Convert CMYK values to RGB.

    Parameters:
        c (float): Cyan component (0.0 to 1.0)
        m (float): Magenta component (0.0 to 1.0)
        y (float): Yellow component (0.0 to 1.0)
        k (float): Black key component (0.0 to 1.0)

    Returns:
        tuple: RGB values as integers (0 to 1)
    """
    if any(value < 0.0 or value > 1.0 for value in [c, m, y, k]):
        raise ValueError("CMYK values must be between 0.0 and 1.0.")

    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)

    return int(r), int(g), int(b)


if __name__ == '__main__':
    file_exists = Path(data_file_name).exists()
    # creating training-cases
    x_train = np.random.random((batch_size, 4))
    y_train = np.array([np.array(cmyk_to_rgb(*x)) for x in x_train])

    # creating test-cases
    x_test = np.random.random((test_cases, 4))
    y_test = np.array([np.array(cmyk_to_rgb(*x)) for x in x_test])

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
        eval_every=eval_every
    )

    # Save the trained network
    save_to_file(neural_network, data_file_name)





