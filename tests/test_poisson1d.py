from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_line_mesh, build_complex_from_mesh
from dctkit.math.opt import optctrl as oc
from deap import gp
from alpine.gp import gpsymbreg as gps
from dctkit import config
import dctkit
import numpy as np
import ray
import math
import yaml
from typing import Tuple, Callable, List
import numpy.typing as npt
import os
import pytest

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


def eval_MSE_sol(residual: Callable, X: npt.NDArray, y: npt.NDArray,
                 S: SimplicialComplex, u_0: C.CochainP0) -> float:

    num_nodes = X.shape[1]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # objective: squared norm of the residual of the equation + penalty on Dirichlet
    # boundary condition on the first node
    def obj(x, y):
        penalty = 100.*x[0]**2
        u = C.CochainP0(S, x)
        f = C.CochainP0(S, y)
        r = residual(u, f)
        total_energy = C.inner_product(r, r) + penalty
        return total_energy

    prb = oc.OptimizationProblem(dim=num_nodes, state_dim=num_nodes, objfun=obj)

    MSE = 0.

    u = []

    for i, vec_y in enumerate(y):
        # set additional arguments of the objective function
        # (apart from the vector of unknowns)
        args = {'y': vec_y}
        prb.set_obj_args(args)

        # minimize the objective
        x = prb.solve(x0=u_0.coeffs, ftol_abs=1e-12, ftol_rel=1e-12, maxeval=1000)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):

            current_err = np.linalg.norm(x-X[i, :])**2
        else:
            current_err = math.nan

        if math.isnan(current_err):
            MSE = 1e5
            break

        MSE += current_err

        u.append(x)

    MSE *= 1/X.shape[0]

    return MSE, u


@ray.remote
def predict(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
            S: SimplicialComplex, u_0: C.CochainP0,
            penalty: dict) -> List[npt.NDArray]:

    _, u = eval_MSE_sol(individual, X, y, S, u_0)

    return u


@ray.remote
def score(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
          S: SimplicialComplex, u_0: C.CochainP0,
          penalty: dict) -> List[npt.NDArray]:

    MSE, _ = eval_MSE_sol(individual, X, y, S, u_0)

    return MSE


@ray.remote
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 S: SimplicialComplex, u_0: C.CochainP0,
                 penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, X, y, S, u_0)

    # add penalty on length of the tree to promote simpler solutions
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


cases = ['poisson1d_1.yaml', 'poisson1d_2.yaml']


@pytest.mark.parametrize('yamlfile', cases)
def test_poisson1d(set_test_dir, yamlfile):
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    # generate mesh and dataset
    mesh, _ = generate_line_mesh(num_nodes=11, L=1.)
    S = build_complex_from_mesh(mesh)
    S.get_hodge_star()
    x = S.node_coords
    num_nodes = S.num_nodes

    # generate training and test datasets
    # exact solution = x²
    u = C.CochainP0(S, np.array(x[:, 0]**2, dtype=dctkit.float_dtype))
    # compute source term such that u solves the discrete Poisson equation
    # Delta u + f = 0, where Delta is the discrete Laplace-de Rham operator
    f = C.laplacian(u)
    f.coeffs *= -1.
    X_train = np.array([u.coeffs], dtype=dctkit.float_dtype)
    y_train = np.array([f.coeffs], dtype=dctkit.float_dtype)

    # initial guess for the unknown of the Poisson problem (cochain of nodals values)
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    # define primitive set for the residual of the discrete Poisson equation
    pset = gp.PrimitiveSetTyped("RESIDUAL", [C.CochainP0, C.CochainP0], C.CochainP0)

    # rename arguments of the residual
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    seed_str = ["AddCP0(delP1(cobP0(u)),f)"]

    penalty = config_file_data["gp"]["penalty"]
    common_params = {'S': S, 'u_0': u_0, 'penalty': penalty}

    gpsr = gps.GPSymbolicRegressor(
        pset=pset, fitness=eval_fitness.remote,
        error_metric=score.remote, predict_func=predict.remote,
        config_file_data=config_file_data,
        common_data=common_params, feature_extractors=[len],
        seed=seed_str)

    param_names = ('X', 'y')

    gpsr.fit(X_train, y_train, param_names, X_val=X_train, y_val=y_train)

    u_best = gpsr.predict(X_train, y_train, param_names)

    ray.shutdown()
    assert np.allclose(u.coeffs, np.ravel(u_best))
