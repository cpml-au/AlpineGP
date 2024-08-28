import numpy as np
from sklearn.model_selection import train_test_split
from pmlb import fetch_data


def generate_dataset(problem="Nguyen-8"):
    np.random.seed(42)
    range_train = None
    num_variables = 1
    if problem == "Nguyen-1":
        range_train = (-1.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.power(x, 3) + np.power(x, 2) + x

    elif problem == "Nguyen-2":
        range_train = (-1.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.power(x, 4) + np.power(x, 3) + np.power(x, 2) + x

    elif problem == "Nguyen-3":
        range_train = (-1.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.power(x, 5) + np.power(x, 4) + np.power(x, 3) + np.power(x, 2) + x

    elif problem == "Nguyen-4":
        range_train = (-1.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return (
                np.power(x, 6)
                + np.power(x, 5)
                + np.power(x, 4)
                + np.power(x, 3)
                + np.power(x, 2)
                + x
            )

    elif problem == "Nguyen-5":
        range_train = (-1.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.sin(x * x) * np.cos(x) - 1.0

    elif problem == "Nguyen-6":
        range_train = (-1.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.sin(x) + np.sin(x + x * x)

    elif problem == "Nguyen-7":
        range_train = (1.0e-3, 2.0, 20)
        range_test = (3.0, 5.0, 20)

        def func(x):
            return np.log(1 + x) + np.log(x * x + 1)

    elif problem == "Nguyen-8":
        range_train = (1.0e-3, 4.0, 20)
        range_test = (4.0, 8.0, 20)
        func = np.sqrt
    elif problem == "Nguyen-9":
        num_variables = 2
        range_train = (0.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.sin(x[:, 0]) + np.sin(x[:, 1] * x[:, 1])

    elif problem == "Nguyen-10":
        num_variables = 2
        range_train = (0.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return 2.0 * np.sin(x[:, 0]) * np.cos(x[:, 1])

    elif problem == "Nguyen-11":
        num_variables = 2
        range_train = (0.0, 1.0, 20)
        range_test = (1.0, 3.0, 20)

        def func(x):
            return np.power(x[:, 0], x[:, 1])

    elif problem == "Nguyen-12":
        num_variables = 2
        range_train = (-3.0, 3.0, 20)
        range_test = (0.0, 1.0, 20)

        def func(x):
            return (
                np.power(x[:, 0], 4)
                - np.power(x[:, 0], 3)
                + 0.5 * np.power(x[:, 1], 2)
                - x[:, 1]
            )

    elif problem == "Nguyen-13":
        range_train = (-1.0, 1.0, 20)
        range_test = (0.0, 1.0, 20)

        def func(x):
            return 3.39 * np.power(x, 3) + 2.12 * np.power(x, 2) + 1.78 * x

    # Nguyen datasets
    if num_variables > 1 and range_train is not None:
        X_train = np.empty((range_train[-1], num_variables))
        X_test = np.empty((range_test[-1], num_variables))
        for i in range(num_variables):
            X_train[:, i] = np.random.uniform(*range_train)
            X_test[:, i] = np.random.uniform(*range_test)
    elif num_variables == 1 and range_train is not None:
        X_train = np.random.uniform(*range_train)
        X_test = np.random.uniform(*range_test)

    if range_train is not None:
        y_train = func(X_train)
        y_test = func(X_test)
    else:
        if problem == "C1":
            # Cosmo dataset from https://github.com/CP3-Origins/Things-to-bench
            data = np.loadtxt("C1.csv", delimiter=",", skiprows=1)
            X = data[:, 1]
            y = data[:, 0]
            num_variables = 1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        else:
            # PMLB datasets
            X, y = fetch_data(problem, return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            num_variables = X.shape[1]

    return X_train, y_train, X_test, y_test, num_variables
