import numpy as np
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
import os
from dctkit import config
import alpine.data.util as u
import numpy.typing as npt
from typing import Tuple

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


cwd = os.path.dirname(simplex.__file__)
data_path = os.path.dirname(os.path.realpath(__file__))


def generate_dataset(S: simplex.SimplicialComplex, num_samples_per_source: int,
                     num_sources: int, noise: npt.NDArray) -> Tuple[npt.NDArray,
                                                                    npt.NDArray]:
    """Generate a dataset for the Poisson problem Delta u + f = 0, i.e. collection of
    pairs source term - solution vectors.

    Args:
        S: simplicial complex where the functions of the dataset
            are defined.
        num_samples_per_source: the multiplicity of every class (for now 3) of
            functions of the dataset.
        num_sources: number of types (1-3) of functions used to represent the
            source term.
        noise: noise to perturb the solution vector.

    Returns:
        np.array of the dataset samples.
        np.array of the labels.
    """
    node_coords = S.node_coords
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
            f_i = C.laplacian(u_i_coch).coeffs
            data_X[num_sources*i, :] = u_i + (max(u_i) - min(u_i))*noise
            data_y[num_sources*i, :] = f_i

        if num_sources >= 2:
            # ith exponential function
            u_i = (i+1)*np.log(1 + x) + \
                1/(i+1)*np.log(1 + y)
            u_i_coch = C.CochainP0(S, u_i)
            # f_i = (i+1)/((1 + x)**2) + 1/((i+1)*(1+y)**2)
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


def save_noise(num_nodes: int):
    noise = 1.*np.random.rand(num_nodes)
    np.save("noise_poisson.npy", noise)


def load_noise():
    return np.load(os.path.join(data_path, "noise_poisson.npy"))


if __name__ == '__main__':
    # set seed
    np.random.seed(42)
    mesh, _ = util.generate_square_mesh(0.08)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    num_nodes = S.num_nodes
    data_generator_kwargs = {'S': S, 'num_samples_per_source': 4, 'num_sources': 3,
                             'noise': 0.*np.random.rand(num_nodes)}
    u.save_dataset(data_generator=generate_dataset,
                   data_generator_kwargs=data_generator_kwargs,
                   perc_val=0.25,
                   perc_test=0.25)
