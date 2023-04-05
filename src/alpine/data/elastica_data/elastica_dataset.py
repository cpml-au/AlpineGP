import os
import numpy as np
import dctkit as dt
from alpine.data.poisson_dataset import split_dataset

data_path = os.path.dirname(os.path.realpath(__file__))


def get_data_with_noise(noise):
    # get data
    y = np.array([-5., -10., -15., -20.], dtype=dt.float_dtype)
    X = np.empty((4, 10), dtype=dt.float_dtype)
    data_string = ["xy_F_5.txt", "xy_F_10.txt", "xy_F_15.txt", "xy_F_20.txt"]
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
    X, y = split_dataset(data_X, data_y, 0.25, 0.25, True)
    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y
    np.savetxt("X_train_elastica.csv", X_train, delimiter=",")
    np.savetxt("X_valid_elastica.csv", X_valid, delimiter=",")
    np.savetxt("X_test_elastica.csv", X_test, delimiter=",")
    np.savetxt("y_train_elastica.csv", y_train, delimiter=",")
    np.savetxt("y_valid_elastica.csv", y_valid, delimiter=",")
    np.savetxt("y_test_elastica.csv", y_test, delimiter=",")


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
    X_train = np.loadtxt(os.path.join(data_path, "X_train_elastica.csv"),
                         dtype=float, delimiter=",")
    X_valid = np.loadtxt(os.path.join(data_path, "X_valid_elastica.csv"),
                         dtype=float, delimiter=",")
    X_test = np.loadtxt(os.path.join(data_path, "X_test_elastica.csv"),
                        dtype=float, delimiter=",")
    y_train = np.loadtxt(os.path.join(data_path, "y_train_elastica.csv"),
                         dtype=float, delimiter=",")
    y_valid = np.loadtxt(os.path.join(data_path, "y_valid_elastica.csv"),
                         dtype=float, delimiter=",")
    y_test = np.loadtxt(os.path.join(data_path, "y_test_elastica.csv"),
                        dtype=float, delimiter=",")
    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == '__main__':
    # seet seed
    np.random.seed(42)
    save_data(0.01*np.random.rand(11))
