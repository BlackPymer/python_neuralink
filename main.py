import numpy as np


def logical_transformation(x):
    """
    Определяет логическую зависимость между входом x и выходом y.
    x: вектор размерности (8,)
    Возвращает: вектор y размерности (10,)
    """
    y0 = x[0] + x[1]  # Сумма первых двух элементов
    y1 = x[2] - x[3]  # Разность 3-го и 4-го
    y2 = x[4] * 0.5  # Половина 5-го элемента
    y3 = 1.0 if x[5] > 0.5 else 0.0  # Бинарный порог для 6-го элемента
    y4 = x[6] ** 2  # Квадрат 7-го элемента
    y5 = np.log(x[7] + 1)  # Логарифм (сдвиг на 1, чтобы избежать log(0))
    y6 = (x[0] + x[1] + x[2]) / 3  # Среднее первых трёх
    y7 = (x[3] + x[4] + x[5]) / 3  # Среднее 4-го, 5-го и 6-го
    y8 = (x[6] + x[7]) / 2  # Среднее последних двух
    y9 = x[0] * x[7]  # Произведение первого и последнего
    return np.array([y0, y1, y2, y3, y4, y5, y6, y7, y8, y9])


def generate_deterministic_data(num_samples, input_dim=8):

    X = np.zeros((num_samples, input_dim))
    for i in range(num_samples):
        start = 0.0 + i * 0.01
        end = 1.0 + i * 0.01
        X[i] = np.linspace(start, end, input_dim)
    y = np.array([logical_transformation(x) for x in X])
    return X, y


if __name__ == '__main__':

    from Layers.dense import Dense
    from Loss.mean_squared_error import MeanSquaredError
    from Operations.sigmoid import Sigmoid
    from Optimizers.stochastic_gradient_descent import SGD
    from Trainer import Trainer
    from neural_network import NeuralNetwork


    nn = NeuralNetwork(
        layers=[Dense(neurons=13, activation=Sigmoid()), Dense(neurons=10)],
        loss=MeanSquaredError()
    )

    num_train_samples = 50
    num_test_samples = 20
    input_dim = 8
    output_dim = 10


    X_train, y_train = generate_deterministic_data(num_train_samples, input_dim)
    X_test, y_test = generate_deterministic_data(num_test_samples, input_dim)


    trainer = Trainer(nn, SGD(learning_rate=0.01))


    trainer.fit(X_train, y_train, X_test, y_test,
                epochs=1000,
                eval_every=10,
                seed=20250327)
    print("Training finished")