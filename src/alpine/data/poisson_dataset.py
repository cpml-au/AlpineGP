import numpy as np


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
        e_i = np.exp(i*node_coords[:, 0]) + np.exp(i*node_coords[:, 1])
        rhs_ei = i**2 * e_i

        # ith power function
        p_i = node_coords[:, 0]**(i+2) + node_coords[:, 1]**(i+2)
        rhs_pi = (i+2)*(i+1)*(node_coords[:, 0]**i + node_coords[:, 1]**i)

        # fill the dataset
        data_X[i, :] = q_i
        data_X[i+1, :] = e_i
        data_X[i+2, :] = p_i
        data_y[i, :] = rhs_qi
        data_y[i+1, :] = rhs_ei
        data_y[i+2, :] = rhs_pi

    return data_X, data_y
