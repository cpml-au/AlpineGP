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


def get_LE_boundary_values(X, y, ref_node_coords, boundary_nodes_info):
    bvalues_X = []
    right_bnd_nodes_idx = boundary_nodes_info['right_bnd_nodes_idx']
    left_bnd_nodes_idx = boundary_nodes_info['left_bnd_nodes_idx']
    up_bnd_nodes_idx = boundary_nodes_info['up_bnd_nodes_idx']
    down_bnd_nodes_idx = boundary_nodes_info['down_bnd_nodes_idx']
    for i, data_label in enumerate(y):
        true_curr_node_coords = X[i, :, :]
        if data_label == "pure_tension":
            bot_left_corn_idx = left_bnd_nodes_idx.index(0)
            bottom_left_corner = left_bnd_nodes_idx[bot_left_corn_idx]
            left_bnd_nodes_without_corner = left_bnd_nodes_idx[:bot_left_corn_idx] + \
                left_bnd_nodes_idx[(bot_left_corn_idx+1):]

            left_bnd_pos_components = [0]
            right_bnd_pos_components = [0]

            left_bnd_nodes_pos = ref_node_coords[left_bnd_nodes_without_corner,
                                                 :][:, left_bnd_pos_components]
            bottom_left_corner_pos = ref_node_coords[bottom_left_corner, :]
            right_bnd_nodes_pos = true_curr_node_coords[right_bnd_nodes_idx,
                                                        :][:, right_bnd_pos_components]

            # NOTE: without flatten it does not work properly when concatenating
            # multiple bcs; fix this so that flatten is not needed (not intuitive)
            boundary_values = {"0": (left_bnd_nodes_without_corner+right_bnd_nodes_idx,
                                     np.vstack((left_bnd_nodes_pos,
                                               right_bnd_nodes_pos)).flatten()),
                               ":": (bottom_left_corner, bottom_left_corner_pos)}
        elif data_label == "pure_shear":
            up_bnd_nodes_pos_x = true_curr_node_coords[up_bnd_nodes_idx, 0]
            up_bnd_nodes_pos_y = ref_node_coords[up_bnd_nodes_idx, 1]
            up_bnd_pos = np.zeros((len(up_bnd_nodes_idx), 3))
            up_bnd_pos[:, 0] = up_bnd_nodes_pos_x
            up_bnd_pos[:, 1] = up_bnd_nodes_pos_y
            down_bnd_pos = ref_node_coords[down_bnd_nodes_idx, :]
            left_bnd_pos = true_curr_node_coords[left_bnd_nodes_idx, :]
            right_bnd_pos = true_curr_node_coords[right_bnd_nodes_idx, :]
            bnodes = left_bnd_nodes_idx + right_bnd_nodes_idx + \
                up_bnd_nodes_idx + down_bnd_nodes_idx
            bvalues = np.vstack((left_bnd_pos, right_bnd_pos,
                                 up_bnd_pos, down_bnd_pos))
            boundary_values = {":": (bnodes, bvalues)}

        bvalues_X.append(boundary_values)
    return bvalues_X
