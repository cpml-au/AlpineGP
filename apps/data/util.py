import numpy as np
import os
from typing import Tuple, Callable, Dict
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from functools import partial

# FIXME: FIX THE DOCS


def split_dataset(X, y, perc_val, perc_test, shuffle=True):
    """Split the dataset in training, validation and test set (double hold out).
    Args:
        X (np.array): samples of the dataset
        y (np.array): targets of the dataset
        perc_val (float): percentage of the dataset dedicated to validation set
        perc_test (float): percentage of the dataset dedicated to validation set

    Returns:
        (tuple): tuple of training and test samples.
        (tuple): tuple of training and test targets.
    """

    # split the dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=perc_test, random_state=42, shuffle=shuffle)

    # split X_train in training and validation set

    X_t, X_valid, y_t, y_valid = train_test_split(
        X_train, y_train, test_size=perc_val, random_state=42, shuffle=shuffle)

    X = (X_t, X_valid, X_test)
    y = (y_t, y_valid, y_test)

    return X, y


def save_dataset(data_generator: Callable, data_generator_kwargs: Dict,
                 perc_val: float, perc_test: float, format: str = "csv", shuffle=True):
    """Generate, split and save the dataset.

    Args:
        S (SimplicialComplex): simplicial complex where the functions of the dataset
        are defined.
        num_samples_per_source (int): the multiplicity of every class (for now 3) of
        functions of the dataset.
        num_sources (int): number of types (1-3) of functions used to represent the
        source term.
        different functions in the dataset.
        noise (np.array): noise to perturb the solution vector.
    """
    data_X, data_y = data_generator(**data_generator_kwargs)
    X, y = split_dataset(data_X, data_y, perc_val, perc_test, shuffle)
    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y
    if format == "csv":
        savefunc = partial(np.savetxt, delimiter=",")
    elif format == "npy":
        savefunc = np.save
    savefunc("X_train." + format, X_train)
    savefunc("X_valid." + format, X_valid)
    savefunc("X_test." + format, X_test)
    savefunc("y_train." + format, y_train)
    savefunc("y_valid." + format, y_valid)
    savefunc("y_test." + format, y_test)


def load_dataset(data_path: str, format: str = "csv") -> Tuple[npt.NDArray]:
    """Load the dataset from .csv files.

    Returns:
        (np.array): training samples.
        (np.array): validation samples.
        (np.array): test samples.
        (np.array): training targets.
        (np.array): validation targets.
        (np.array): test targets.


    """
    if format == "csv":
        loadfunc = partial(np.loadtxt, delimiter=",", dtype=float)
    elif format == "npy":
        loadfunc = partial(np.load, allow_pickle=True)
    X_train = loadfunc(os.path.join(data_path, "X_train." + format))
    X_valid = loadfunc(os.path.join(data_path, "X_valid." + format))
    X_test = loadfunc(os.path.join(data_path, "X_test." + format))
    y_train = loadfunc(os.path.join(data_path, "y_train." + format))
    y_valid = loadfunc(os.path.join(data_path, "y_valid." + format))
    y_test = loadfunc(os.path.join(data_path, "y_test." + format))
    return X_train, X_valid, X_test, y_train, y_valid, y_test
