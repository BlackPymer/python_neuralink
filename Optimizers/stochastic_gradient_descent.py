import numpy as np

from Optimizers import optimizer


class SGD(optimizer.Optimizer):
    """
    Stochastic gradient optimizer
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__(learning_rate)

    def step(self) -> None:
        """
        For every param
        """

        for i, (param, param_grad) in enumerate(zip(self.net.params(), self.net.param_grads())):
            #print(
            #   f"Before update - Layer {i}: param norm = {np.linalg.norm(param)}, grad norm = {np.linalg.norm(param_grad)}")
            param -= self.learning_rate * param_grad
            #print(f"After update - Layer {i}: param norm = {np.linalg.norm(param)}")
