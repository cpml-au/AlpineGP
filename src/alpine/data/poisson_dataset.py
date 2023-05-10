import numpy as np
# import multiprocessing
import gmsh
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
import os
from sklearn.model_selection import train_test_split
from matplotlib import tri
import matplotlib.pyplot as plt


cwd = os.path.dirname(simplex.__file__)
data_path = os.path.dirname(os.path.realpath(__file__))


def generate_complex(lc):
    """Generate a Simplicial complex and its boundary nodes from a msh file.

    Args:
        lc (float): target mesh file.

    Returns:
        (SimplicialComplex): resulting simplicial complex.
        (np.array): np.array containing the positions of the boundary nodes.
    """

    _, _, S_2, node_coords = util.generate_square_mesh(lc)

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
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    # source term data
    data_X = np.empty((num_sources*num_samples_per_source, num_nodes))
    # solution data
    data_y = np.empty((num_sources*num_samples_per_source, num_nodes))

    for i in range(num_samples_per_source):
        if num_sources >= 1:
            # ith quadratic function
            u_i = (i+1)*np.exp(np.sin(x)) + ((i+1)**2)*np.exp(np.cos(y))
            u_i_coch = C.CochainP0(S, u_i)
            #f_i = -(i+1)*(np.exp(np.sin(x))*(np.cos(x))
            #              ** 2 - np.exp(np.sin(x))*np.sin(x)) + ((i+1)**2)*(-np.exp(np.cos(y))*(np.sin(y))**2 +
            #                                                                np.exp(np.cos(y))*np.cos(y))
            f_i = C.laplacian(u_i_coch).coeffs
            data_X[num_sources*i, :] = u_i + (max(u_i) - min(u_i))*noise
            data_y[num_sources*i, :] = f_i

        if num_sources >= 2:
            # ith exponential function
            u_i = (i+1)*np.log(1 + x) + \
                1/(i+1)*np.log(1 + y)
            u_i_coch = C.CochainP0(S, u_i)
            #f_i = (i+1)/((1 + x)**2) + 1/((i+1)*(1+y)**2)
            f_i = C.laplacian(u_i_coch).coeffs
            data_X[num_sources*i+1, :] = u_i + (max(u_i) - min(u_i))*noise
            data_y[num_sources*i+1, :] = f_i

        if num_sources >= 3:
            # ith power function
            u_i = x**(i+3) + y**(i+3)
            u_i_coch = C.CochainP0(S, u_i)
            f_i = -(i+3)*(i+2)*(x**(i+1) + y**(i+1))
            f_i = C.laplacian(u_i_coch).coeffs
            data_X[num_sources*i+2, :] = u_i + (max(u_i) - min(u_i))*noise
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
    """

    if not is_valid:
        return X, y

    # split the dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=perc_test, random_state=42)

    # split X_train in training and validation set

    X_t, X_valid, y_t, y_valid = train_test_split(
        X_train, y_train, test_size=perc_val, random_state=42)

    X = (X_t, X_valid, X_test)
    y = (y_t, y_valid, y_test)

    return X, y


def save_dataset(S, num_samples_per_source, num_sources, noise):
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
    data_X, data_y = generate_dataset(S, num_samples_per_source, num_sources, noise)
    X, y = split_dataset(data_X, data_y, 0.25, 0.25, True)
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


def save_noise(num_nodes):
    noise = 1.*np.random.rand(num_nodes)
    np.save("noise_poisson.npy", noise)


def load_noise():
    return np.load(os.path.join(data_path, "noise_poisson.npy"))


if __name__ == '__main__':
    # seet seed
    np.random.seed(42)
    S, bnodes, triang = generate_complex(0.08)
    num_nodes = S.num_nodes
    save_dataset(S, 4, 3, 0.*np.random.rand(num_nodes))
    # save_noise(num_nodes)
    # data_X, _ = generate_dataset(S, 4, 3, 0.*np.random.rand(num_nodes))
    # for i in range(12):
    #    plt.tricontourf(triang, data_X[i, :], cmap='RdBu', levels=20)
    #    plt.show()
