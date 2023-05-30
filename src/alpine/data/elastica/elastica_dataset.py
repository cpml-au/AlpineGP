import os
import numpy as np
import dctkit as dt
from alpine.data.poisson.poisson_dataset import split_dataset

data_path = os.path.dirname(os.path.realpath(__file__))


def get_data_with_noise(noise):
    # get data
    abs_force = 5*np.arange(1, 11)
    y = -abs_force.astype(dt.float_dtype)
    X = np.empty((10, 10), dtype=dt.float_dtype)
    data_string = ["xy_F_" + str(i) + ".txt" for i in abs_force]
    for i, string in enumerate(data_string):
        data = np.loadtxt(os.path.join(data_path, string), dtype=float)
        # get data position
        x_data = data[:, 1][::10] + noise
        y_data = data[:, 2][::10] + noise
        # get theta
        theta_true = np.empty(10, dtype=dt.float_dtype)
        for j in range(10):
            theta_true[j] = np.arctan(
                (y_data[j+1]-y_data[j])/(x_data[j+1]-x_data[j]))
        # update X
        X[i, :] = theta_true
    return X, y


def save_data(noise):
    data_X, data_y = get_data_with_noise(noise)
    X, y = split_dataset(data_X, data_y, 0.3, 0.2, True)
    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y
    np.savetxt("X_train.csv", X_train, delimiter=",")
    np.savetxt("X_valid.csv", X_valid, delimiter=",")
    np.savetxt("X_test.csv", X_test, delimiter=",")
    np.savetxt("y_train.csv", y_train, delimiter=",")
    np.savetxt("y_valid.csv", y_valid, delimiter=",")
    np.savetxt("y_test.csv", y_test, delimiter=",")


def load_dataset():
    """Load the dataset from .csv files.

    Returns:
        (np.array): training samples.
        (np.array): validation samples.
        (np.array): test samples.
        (np.array): training targets.
        (np.array): validation targets.
        (np.array): test targets.


    """
    X_train = np.loadtxt(os.path.join(data_path, "X_train.csv"),
                         dtype=float, delimiter=",")
    X_valid = np.loadtxt(os.path.join(data_path, "X_valid.csv"),
                         dtype=float, delimiter=",")
    X_test = np.loadtxt(os.path.join(data_path, "X_test.csv"),
                        dtype=float, delimiter=",")
    y_train = np.loadtxt(os.path.join(data_path, "y_train.csv"),
                         dtype=float, delimiter=",")
    y_valid = np.loadtxt(os.path.join(data_path, "y_valid.csv"),
                         dtype=float, delimiter=",")
    y_test = np.loadtxt(os.path.join(data_path, "y_test.csv"),
                        dtype=float, delimiter=",")
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == '__main__':
    # seet seed
    np.random.seed(42)
    save_data(0.005*np.random.rand(11))
