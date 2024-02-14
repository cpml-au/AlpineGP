from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from matplotlib import tri
from deap import gp, base
from alpine.data.util import load_dataset
import alpine.data.poisson.poisson_dataset as pd
from dctkit.mesh import util
from alpine.gp import gpsymbreg as gps
from dctkit import config
import dctkit

import ray

import numpy as np
import jax.numpy as jnp
import math
import time
import sys
import yaml
from typing import Tuple, Callable
import numpy.typing as npt

residual_formulation = False

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()

noise = pd.load_noise()


def is_valid_energy(u: npt.NDArray, prb: oc.OptimizationProblem,
                    bnodes: npt.NDArray) -> bool:
    # perturb solution and check whether the gradient still vanishes
    # (i.e. constant energy)
    u_noise = u + noise*np.mean(u)
    u_noise[bnodes] = u[bnodes]
    grad_u_noise = jnp.linalg.norm(prb.solver.gradient(u_noise))
    is_valid = grad_u_noise >= 1e-6
    return is_valid


def eval_MSE_sol(func: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
                 gamma: float, u_0: C.CochainP0) -> Tuple[float, npt.NDArray]:

    num_nodes = X.shape[1]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(x, vec_y, vec_bvalues):
        penalty = 0.5*gamma*jnp.sum((x[bnodes] - vec_bvalues)**2)
        c = C.CochainP0(S, x)
        fk = C.CochainP0(S, vec_y)
        if residual_formulation:
            total_energy = C.inner_product(func(c, fk), func(c, fk)) + penalty
        else:
            total_energy = func(c, fk) + penalty
        return total_energy

    prb = oc.OptimizationProblem(
        dim=num_nodes, state_dim=num_nodes, objfun=total_energy)

    total_err = 0.

    best_sols = []

    for i, vec_y in enumerate(y):
        # extract current bvalues
        vec_bvalues = bvalues[i, :]

        # set current bvalues and vec_y for the Poisson problem
        args = {'vec_y': vec_y, 'vec_bvalues': vec_bvalues}
        prb.set_obj_args(args)

        # minimize the objective
        x = prb.solve(x0=u_0.coeffs, ftol_abs=1e-12, ftol_rel=1e-12, maxeval=1000)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):
            # check whether the energy is "admissible" (i.e. exclude constant energies)
            valid_energy = is_valid_energy(u=x, prb=prb, bnodes=bnodes)

            if valid_energy:
                current_err = np.linalg.norm(x-X[i, :])**2
            else:
                current_err = math.nan
        else:
            current_err = math.nan

        if math.isnan(current_err):
            total_err = 1e5
            break

        total_err += current_err

        best_sols.append(x)

    total_err *= 1/X.shape[0]

    return total_err, best_sols


@ray.remote(num_cpus=2)
def eval_best_sols(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                   bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
                   gamma: float, u_0: npt.NDArray, penalty: dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(individual, indlen, X, y, bvalues, S, bnodes,
                                gamma, u_0)

    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
             bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: npt.NDArray, penalty: dict) -> float:

    MSE, _ = eval_MSE_sol(individual, indlen, X, y, bvalues, S, bnodes, gamma, u_0)

    return MSE


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray, gamma: float,
                 u_0: npt.NDArray, penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, indlen, X, y, bvalues, S, bnodes,
                                gamma, u_0)

    # penalty terms on length
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


# Plot best solution
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: C.CochainP0, toolbox: base.Toolbox,
             triang: tri.Triangulation):

    indfun = toolbox.compile(expr=ind)

    _, u = eval_MSE_sol(indfun, indlen=0, X=X, y=y, bvalues=bvalues, S=S,
                        bnodes=bnodes, gamma=gamma, u_0=u_0)

    plt.figure(10, figsize=(8, 4))
    plt.clf()
    fig = plt.gcf()
    _, axes = plt.subplots(2, 3, num=10)
    for i in range(0, 3):
        axes[0, i].tricontourf(triang, u[i], cmap='RdBu', levels=20)
        pltobj = axes[1, i].tricontourf(triang, X[i], cmap='RdBu', levels=20)
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_box_aspect(1)
    plt.colorbar(pltobj, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


def stgp_poisson(config_file, output_path=None):
    global residual_formulation
    # generate mesh and dataset
    mesh, _ = util.generate_square_mesh(0.08)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    bnodes = mesh.cell_sets_dict["boundary"]["line"]
    num_nodes = S.num_nodes
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(pd.data_path, "csv")

    # extract boundary values
    bvalues_train = X_train[:, bnodes]
    bvalues_val = X_val[:, bnodes]
    bvalues_test = X_test[:, bnodes]

    # penalty parameter for the Dirichlet bcs
    gamma = 1000.

    # initial guess for the solution of the Poisson problem
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    residual_formulation = config_file["gp"]["residual_formulation"]

    # define primitive set and add primitives and terminals
    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], C.Cochain)
        # ones cochain
        pset.addTerminal(C.Cochain(S.num_nodes, True, S, np.ones(
            S.num_nodes, dtype=dctkit.float_dtype)), C.Cochain, name="ones")
    else:
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float)

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")
    # rename arguments
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbolicRegressor(pset=pset, config_file_data=config_file)

    penalty = config_file["gp"]["penalty"]

    GPprb.store_eval_common_params({'S': S, 'penalty': penalty,
                                    'bnodes': bnodes,
                                    'gamma': gamma, 'u_0': u_0})

    params_names = ('X', 'y', 'bvalues')
    datasets = {'train': [X_train, y_train, bvalues_train],
                'val': [X_val, y_val, bvalues_val],
                'test': [X_test, y_test, bvalues_test]}
    GPprb.store_datasets_params(params_names, datasets)

    GPprb.register_eval_funcs(fitness=eval_fitness.remote, error_metric=eval_MSE.remote,
                              eval_sol=eval_best_sols.remote)

    if GPprb.plot_best:
        triang = tri.Triangulation(S.node_coords[:, 0], S.node_coords[:, 1], S.S[2])
        GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val, y=y_val,
                               bvalues=bvalues_val, S=S, bnodes=bnodes,
                               gamma=gamma, u_0=u_0,
                               toolbox=GPprb.toolbox, triang=triang)

    GPprb.__register_map([len])

    start = time.perf_counter()
    # opt_string = "SquareF(InnP0(InvMulP0(u, InnP0(u, fk)), delP1(dP0(u))))"
    # opt_string = "SubF(InnP1(cobP0(u), cobP0(u)), MulF(2., InnP0(f, u)))"
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPprb.__run(print_log=True, seed=None,
                save_best_individual=True, save_train_fit_history=True,
                save_best_test_sols=True, X_test_param_name="X",
                output_path=output_path)

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))

    # path for output data speficified
    if n_args >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "."

    stgp_poisson(config_file, output_path)
