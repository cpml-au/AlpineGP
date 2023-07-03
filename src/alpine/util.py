from typing import Tuple
from scipy.linalg import block_diag
import numpy as np
import dctkit as dt
from scipy import sparse


# TODO: find a way to avoid recomputing transform (encapsulate this function
# within a class). Also, this function looks too complex...
def get_positions_from_angles(angles: Tuple) -> Tuple:
    """Get x,y coordinates given a tuple containing all theta matrices.
    To do it, we have to solve two linear systems Ax = b_x, Ay = b_y,
    where A is a block diagonal matrix where each block is bidiagonal.

    Args:
        X (tuple): tuple containing theta to transform in coordinates.
        transform (np.array): matrix of the linear system.

    Returns:
        (list): list of x-coordinates
        (list): list of y-coordinates
    """
    # bidiagonal matrix to transform theta in (x,y)
    num_nodes = angles[0].shape[1]+1
    diag = [1]*num_nodes
    upper_diag = [-1]*(num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    x_all = []
    y_all = []
    h = 1/angles[0].shape[1]
    for i in range(len(angles)):
        theta = angles[i]
        dim = theta.shape[0]

        # compute cos and sin theta
        cos_theta = h*np.cos(theta)
        sin_theta = h*np.sin(theta)
        b_x = np.zeros((theta.shape[0], theta.shape[1]+1), dtype=dt.float_dtype)
        b_y = np.zeros((theta.shape[0], theta.shape[1]+1), dtype=dt.float_dtype)
        b_x[:, 1:] = cos_theta
        b_y[:, 1:] = sin_theta
        # reshape to a vector
        b_x = b_x.reshape(theta.shape[0]*(theta.shape[1]+1))
        b_y = b_y.reshape(theta.shape[0]*(theta.shape[1]+1))
        transform_list = [transform]*dim
        T = block_diag(*transform_list)
        # solve the system. In this way we find the solution but
        # as a vector and not as a matrix.
        x_i = np.linalg.solve(T, b_x)
        y_i = np.linalg.solve(T, b_y)
        # reshape again to have a matrix
        x_i = x_i.reshape((theta.shape[0], theta.shape[1]+1))
        y_i = y_i.reshape((theta.shape[0], theta.shape[1]+1))
        # update the list
        x_all.append(x_i)
        y_all.append(y_i)
    return x_all, y_all
