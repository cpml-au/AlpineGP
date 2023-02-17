import numpy as np
# import multiprocessing
import gmsh
from dctkit.mesh import simplex, util
import os
from sklearn.model_selection import train_test_split
from matplotlib import tri

cwd = os.path.dirname(simplex.__file__)


def generate_complex(filename):
    """Generate a Simplicial complex and its boundary nodes from a msh file.

    Args:
        filename (str): name of the msh file (with .msh at the end).

    Returns:
        (SimplicialComplex): resulting simplicial complex.
        (np.array): np.array containing the positions of the boundary nodes.
    """
    # full_path = os.path.join(cwd, filename)
    # _, _, S_2, node_coords = util.read_mesh(full_path)
    _, _, S_2, node_coords = util.generate_square_mesh(0.08)

    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])
    S = simplex.SimplicialComplex(S_2, node_coords, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1

    return S, bnodes, triang


def generate_dataset(S, num_samples_per_source, num_sources, noise):
    """Generate a dataset for the Poisson problem Delta u + f = 0, i.e. collection of
    pairs source term - solution vectors.

    Args:
        S (SimplicialComplex): simplicial complex where the functions of the dataset
        are defined.
        num_samples_per_source (int): the multiplicity of every class (for now 3) of
        functions of the dataset.
        num_sources (int): number of types (1-3) of functions used to represent the
        source term.
        different functions in the dataset.
        noise (np.array): noise to perturb the solution vector.

    Returns:
        np.array: np.array of the dataset samples.
        np.array: np.array of the labels.
    """
    node_coords = S.node_coord
    num_nodes = S.num_nodes
    # source term data
    data_X = np.empty((num_sources*num_samples_per_source, num_nodes))
    # solution data
    data_y = np.empty((num_sources*num_samples_per_source, num_nodes))

    for i in range(num_samples_per_source):
        if num_sources >= 1:
            # ith quadratic function
            u_i = 1/(i + 1)**2 * (node_coords[:, 0]**2 + node_coords[:, 1]**2)
            f_i = -(4/(i+1)**2) * np.ones(num_nodes)
            data_X[num_sources*i, :] = u_i + noise
            data_y[num_sources*i, :] = f_i

        if num_sources >= 2:
            # ith exponential function
            u_i = np.cos(i*node_coords[:, 0]) + np.sin(i*node_coords[:, 1])
            f_i = i**2 * u_i
            data_X[num_sources*i+1, :] = u_i + noise
            data_y[num_sources*i+1, :] = f_i

        if num_sources >= 3:
            # ith power function
            u_i = node_coords[:, 0]**(i+2) + node_coords[:, 1]**(i+2)
            f_i = -(i+2)*(i+1)*(node_coords[:, 0]**(i) + node_coords[:, 1]**(i))
            data_X[num_sources*i+2, :] = u_i + noise
            data_y[num_sources*i+2, :] = f_i

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

    X_t, X_valid, y_t, y_valid = train_test_split(
        X_train, y_train, test_size=perc_val, random_state=None)

    X = (X_t, X_valid, X_test)
    y = (y_t, y_valid, y_test)

    return X, y
