class Optimizer(object):
    """
    Basic optimizer class of NN
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Optimizer needs starting learning rate
        """
        self.learning_rate = learning_rate

    def step(self) -> None:
        pass
