from typing import Tuple

import numpy as np
from numpy import ndarray


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def generate_batches(X: ndarray, y: ndarray, size: int = 32) -> Tuple[ndarray]:
    """
    Generates batches for training
    """

    assert X.shape[0] == y.shape[0], \
        '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

    N = X.shape[0]

    for ii in range(0, N, size):
        X_batch, y_batch = X[ii:ii + size], y[ii:ii + size]

        yield X_batch, y_batch
