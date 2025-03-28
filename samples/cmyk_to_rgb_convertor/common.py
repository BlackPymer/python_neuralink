import pickle
from copy import deepcopy

from neural_network import NeuralNetwork

"""contains some variables and functions for bath train and common"""

data_file_name = "data.pkl"


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
    if any(value < 0.0 or value > 1.0 for value in [c, m, y, k]):
        raise ValueError("CMYK values must be between 0.0 and 1.0.")

    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)

    return int(r * 255), int(g * 255), int(b * 255)



def rgb_to_hex(r, g, b):
    """
    Convert RGB values to HEX.

    Parameters:
        r, g, b (int): RGB values (0 to 255)

    Returns:
        str: Hex color code
    """
    return f'#{r:02x}{g:02x}{b:02x}'