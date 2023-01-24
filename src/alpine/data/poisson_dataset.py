import numpy as np
# import multiprocessing
import gmsh
from dctkit.mesh import simplex, util
import os
from sklearn.model_selection import train_test_split, KFold


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


def generate_dataset(S, mult):
    """Generate a dataset for the Poisson problem.

    Args:
        S (SimplicialComplex): simplicial complex where the functions of the dataset
        are defined.
        mult (int): the multiplicity of every class (for now 3) of functions of the
        dataset.

    Returns:
        (np.array): np.array of the dataset samples.
        (np.array): np.array of the labels.
    """
    node_coords = S.node_coord
    num_nodes = S.num_nodes
    data_X = np.empty((3*mult, num_nodes))
    data_y = np.empty((3*mult, num_nodes))
    for i in range(mult):
        # ith quadratic function
        q_i = 1/(i + 1)**2 * (node_coords[:, 0]**2 + node_coords[:, 1]**2)
        rhs_qi = (4/(i+1)**2) * np.ones(num_nodes)

        # ith exponential function
        trig_i = np.cos(i*node_coords[:, 0]) + np.sin(i*node_coords[:, 1])
        rhs_trigi = -i**2 * trig_i

        # ith power function
        p_i = node_coords[:, 0]**(i+2) + node_coords[:, 1]**(i+2)
        rhs_pi = (i+2)*(i+1)*(node_coords[:, 0]**(i) + node_coords[:, 1]**(i))

        # fill the dataset
        data_X[3*i, :] = q_i
        data_X[3*i+1, :] = trig_i
        data_X[3*i+2, :] = p_i
        data_y[3*i, :] = rhs_qi
        data_y[3*i+1, :] = rhs_trigi
        data_y[3*i+2, :] = rhs_pi

    return data_X, data_y


def split_dataset(S, bnodes, num_per_data, k):
    """Split the dataset in training and test set (hold out) and initialize k-fold
    cross validation.

    Args:
        S (SimplicialComplex): a simplicial complex.
        bnodes (np.array): np.array containing the positions of the boundary nodes.
        num_per_data (int): 1/3 of the size of the dataset
        k (int): number of folds for cross validation

    Returns:
        (np.array): training samples
        (np.array): test samples
        (np.array): training labels
        (np.array): test labels
        (KFold): KFold class initialized
    """
    data_X, data_y = generate_dataset(S, num_per_data)

    # split the dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.33, random_state=None)

    # initialize KFOLD
    kf = KFold(n_splits=k, random_state=None)

    return X_train, X_test, y_train, y_test, kf
