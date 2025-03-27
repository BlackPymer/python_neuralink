from copy import deepcopy
from helper_functions import *

from Optimizers.optimizer import Optimizer
from neural_network import NeuralNetwork


class Trainer(object):
    """
    Trains a neural network
    """

    def __init__(self,
                 net: NeuralNetwork,
                 optim: Optimizer) -> None:
        """
        Requires a neural network and an optimizer in order for training to occur.
        Assign the neural network as an instance variable to the optimizer.
        """
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        setattr(self.optim, 'net', self.net)

    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True) -> None:
        """
        Fits the neural network on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluated the neural network on the testing data.
        :param X_train: ndarray of inputs
        :param y_train: ndarray of expected outputs
        :param X_test: input of a case to test the accuracy of trained NN
        :param y_test: expected output of a case to test the accuracy of trained NN
        :param epochs: number of times to train neuralink with the batch
        :param eval_every: every "eval_every" epochs, it evaluated the neural network on the testing data
        :param batch_size: number of cases to train NN with
        :param seed: seed of random generation
        :param restart: reset NN and start training from the very beginning
        :return: None
        """

        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1e9
        last_model = None
        for e in range(epochs):

            if (e + 1) % eval_every == 0:
                # for early stopping
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = generate_batches(X_train, y_train,
                                               batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)

                self.optim.step()

            if (e + 1) % eval_every == 0:

                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e + 1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(
                        f"""Loss increased after epoch {e + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {e + 1 - eval_every}""")
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break

