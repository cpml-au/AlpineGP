import numpy as np
# import multiprocessing
import gmsh
from dctkit.mesh import simplex, util
import os
from sklearn.model_selection import train_test_split


cwd = os.path.dirname(simplex.__file__)


def generate_complex(filename):
    """Generate a Simplicial complex and its boundary nodes from a msh file.

    Args:
        filename (str): name of the msh file (with .msh at the end).

    Returns:
        (SimplicialComplex): resulting simplicial complex.
        (np.array): np.array containing the positions of the boundary nodes.
    """
    full_path = os.path.join(cwd, filename)
    _, _, S_2, node_coords = util.read_mesh(full_path)

    S = simplex.SimplicialComplex(S_2, node_coords)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1

    return S, bnodes


def generate_dataset(S, mult, diff, noise):
    """Generate a dataset for the Poisson problem.

    Args:
        S (SimplicialComplex): simplicial complex where the functions of the dataset
        are defined.
        mult (int): the multiplicity of every class (for now 3) of functions of the
        dataset.
        diff (int): integer (from 1 to 3) that expresses the number of classes of
        different functions in the dataset.
        noise (np.array): noise to perturb data

    Returns:
        (np.array): np.array of the dataset samples.
        (np.array): np.array of the labels.
    """
    node_coords = S.node_coord
    num_nodes = S.num_nodes
    data_X = np.empty((diff*mult, num_nodes))
    data_y = np.empty((diff*mult, num_nodes))
    for i in range(mult):
        if diff >= 1:
            # ith quadratic function
            q_i = 1/(i + 1)**2 * (node_coords[:, 0]**2 + node_coords[:, 1]**2)
            rhs_qi = (4/(i+1)**2) * np.ones(num_nodes)
            data_X[diff*i, :] = q_i + noise
            data_y[diff*i, :] = rhs_qi

        if diff >= 2:
            # ith exponential function
            trig_i = np.cos(i*node_coords[:, 0]) + np.sin(i*node_coords[:, 1])
            rhs_trigi = -i**2 * trig_i
            data_X[diff*i+1, :] = trig_i + noise
            data_y[diff*i+1, :] = rhs_trigi

        if diff >= 3:
            # ith power function
            p_i = node_coords[:, 0]**(i+2) + node_coords[:, 1]**(i+2)
            rhs_pi = (i+2)*(i+1)*(node_coords[:, 0]**(i) + node_coords[:, 1]**(i))
            data_X[diff*i+2, :] = p_i + noise
            data_y[diff*i+2, :] = rhs_pi

    return data_X, data_y


def split_dataset(X, y, perc_val, perc_test, is_valid=False):
    """Split the dataset in training, validation and test set (double hold out).
    Args:
        X (np.array): samples of the dataset
        y (np.array): targets of the dataset
        perc_val (float): percentage of the dataset dedicated to validation set
        perc_test (float): percentage of the dataset dedicated to validation set
        is_valid (bool): boolean that it is True if we want to do model selection
        (validation process).

    Returns:
        (tuple): tuple of training and test samples.
        (tuple): tuple of training and test targets.
        (KFold): KFold class initialized.
    """

    if not is_valid:
        return X, y

    # split the dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=perc_test, random_state=None)

    # split X_train in training and validation set

    X_t, y_t, X_valid, y_valid = train_test_split(
        X_train, y_train, test_size=perc_val, random_state=None)

    X = (X_t, X_valid, X_test)
    y = (y_t, y_valid, y_test)

    return X, y
