# Python Neuralink Library

This repository contains the **Python Neuralink Library** along with sample applications to help you build and train neural networks with ease.

---

## Table of Contents

- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Layers](#layers)
- [Operations](#operations)
- [Training the Neural Network](#training-the-neural-network)
- [Running Sample Applications](#running-sample-applications)
- [Contributing](#contributing)
- [License](#license)

---

## Getting Started

1. **Clone the Repository**  
   Clone this repository into your project directory:
   ```bash
   git clone <repository-url>
2. **Import the Library** In your Python project, import the NeuralNetwork class from neural_network.py:
python

## How it works
The `NeuralNetwork` class automatically initializes and connects layers during forward and backward propagation. This automation allows simple users to utilize the library without needing to understand the intricate details of the underlying propagation process.
### Layers
Layers manage operations and compute gradients during training. You can find these implementations in the `layers` folder. There are two main types:
    1. Layer A basic class meant to be extended for custom layers.
    2. Dense A specialized layer that combines several key operations:
        WeightsMultiply
        BiasAdd
        An optional activation function (e.g., Sigmoid)

### Operations
Operations, located in the operations folder, are responsible for the forward computations and backward gradient calculations. The current operations include:

1.    Operation A fundamental operation class that is intended to be extended.

2.    ParamOperation Inherits from Operation; this class adds the ability to work with parameters and compute gradients.

3.    WeightsMultiply Inherits from ParamOperation; creates weight matrices and multiplies them with the input matrix.

4.    BiasAdd Inherits from ParamOperation; creates a bias matrix and adds it to the input matrix, typically following WeightsMultiply.

5.    Sigmoid Inherits from Operation; applies the sigmoid activation function on the input matrix—usually used as the activation function in a Dense layer.

## Visual Overview
Here’s an ASCII diagram illustrating the key components and their relationships:
```
          +----------------------+
          |  NeuralNetwork       |
          +----------+-----------+
                     |
         [Initializes Layers]
                     |
           +-----------------+
           |   Layers        | <-- (Located in the `layers` folder)
           +--------+--------+
                    /  \
                   /    \
         +--------+      +--------+
         |  Layer |      |  Dense |
         +--------+      +--------+
                             |
                       +-----------+
                       | Operations|
                       |  (Folder) |
                       +-----------+
                             |
        +-------------------------------+
        |  1) Operation                |
        |  2) ParamOperation           |
        |  3) WeightsMultiply          |
        |  4) BiasAdd                  |
        |  5) Sigmoid                  |
        +-------------------------------+
```
## Training the neuralink
To train the network, refer to `trainer.py`:

    Process: The Trainer class handles the generation of training batches and adjusts the network’s parameters until the loss stabilizes.

    Customization: Modify the training parameters to experiment with and optimize the training process according to your needs.

## Running Sample Applications
1. Train the Model: Run `train.py` to start training the Neuralink model.
2. Test the Model: After training, execute `test.py` to evaluate the network with your own test cases.

Thanks Seth Weidman for giving the idea and basic knowledge in this sphere!
