import yaml
import os
import sys
from scipy import sparse
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh
import dctkit as dt
from alpine.gp import gpsymbreg as gps
from alpine.models.elastica import pset
from alpine.data.elastica_data import elastica_dataset as ed
import numpy as np
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)


def elastica_img_from_string(config_file: dict, string: str, X: np.array, y: np.array):
    from stgp_elastica import plot_sol
    # get normalized simplicial complex
    S_1, x = generate_1_D_mesh(num_nodes=11, L=1.)
    S = SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    # bidiagonal matrix to transform theta in (x,y)
    diag = [1]*(S.num_nodes)
    upper_diag = [-1]*(S.num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    # define internal cochain
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0.
    internal_vec[-1] = 0.
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    # add it as a terminal
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")

    # initial guess for the solution
    theta_0 = 0.1*np.random.rand(S.num_nodes-2).astype(dt.float_dtype)

    # initialize toolbox and creator
    createIndividual, toolbox = gps.creator_toolbox_config(
        config_file=config_file, pset=pset)

    ind = createIndividual.from_string(string, pset)
    plot_sol(ind, X, y, toolbox, S, theta_0, transform, False)


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))
    X_train, X_val, X_test, y_train, y_val, y_test = ed.load_dataset()
    string = " Sub(InnD0(theta, delD1(CochMulD1(dD0(theta), St0(int_coch)))), InnD0(FL2_EI0, SinD0(SinD0(theta))))"
    elastica_img_from_string(config_file, string=string, X=X_train, y=y_train)
