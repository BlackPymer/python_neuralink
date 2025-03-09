import numpy as np
from Operations.weight_multiply import WeightMultiply

if __name__ == '__main__':
    weights = np.array([[4,5,6],[1,2,3]])
    print(weights.T)
    a = WeightMultiply(weights=weights.T)
    print(a.forward(np.array([[1,2,3]])))
    print(a.backward(np.array([[1,1]])))