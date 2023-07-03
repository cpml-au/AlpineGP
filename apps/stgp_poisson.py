from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from matplotlib import tri
from deap import gp, base
from alpine.data.poisson import poisson_dataset as pd
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
    grad_u_noise = jnp.linalg.norm(prb.gradient(u_noise))
    is_valid = grad_u_noise >= 1e-6
    return is_valid


def eval_MSE(energy_func: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
             bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: C.CochainP0, return_best_sol: bool = False) -> float:

    num_nodes = X.shape[1]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(x, vec_y, vec_bvalues):
        penalty = 0.5*gamma*jnp.sum((x[bnodes] - vec_bvalues)**2)
        c = C.CochainP0(S, x)
        fk = C.CochainP0(S, vec_y)
        total_energy = energy_func(c, fk) + penalty
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
        x = prb.run(x0=u_0.coeffs, ftol_abs=1e-12, ftol_rel=1e-12, maxeval=1000)

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

    if return_best_sol:
        return best_sols

    total_err *= 1/X.shape[0]

    return total_err


@ray.remote(num_cpus=2)
def eval_MSE_remote(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                    bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
                    gamma: float, u_0: npt.NDArray, penalty: dict) -> Tuple[float, ]:

    total_err = eval_MSE(individual, indlen, X, y, bvalues, S, bnodes, gamma, u_0)

    return total_err,


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray, gamma: float,
                 u_0: npt.NDArray, penalty: dict) -> Tuple[float, ]:

    total_err = eval_MSE(individual, indlen, X, y, bvalues, S, bnodes, gamma, u_0)

    # penalty terms on length
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


# Plot best solution
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: C.CochainP0, toolbox: base.Toolbox,
             triang: tri.Triangulation):

    indfun = toolbox.compile(expr=ind)

    u = eval_MSE(indfun, indlen=0, X=X, y=y, bvalues=bvalues, S=S,
                 bnodes=bnodes, gamma=gamma, u_0=u_0, return_best_sol=True)

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
    # generate mesh and dataset
    mesh = util.generate_square_mesh(0.08)
    S = util.build_complex_from_mesh(mesh)
    num_nodes = S.num_nodes
    X_train, X_val, X_test, y_train, y_val, y_test = pd.load_dataset()

    # extract boundary values
    bvalues_train = X_train[:, bnodes]
    bvalues_val = X_val[:, bnodes]
    bvalues_test = X_test[:, bnodes]

    # penalty parameter for the Dirichlet bcs
    gamma = 1000.

    # initial guess for the solution of the Poisson problem
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    # define primitive set and add primitives and terminals
    pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float)
    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")
    # rename arguments
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file)

    penalty = config_file["gp"]["penalty"]

    # store shared objects refs
    GPprb.store_eval_train_params('common', {'S': S, 'penalty': penalty,
                                             'bnodes': bnodes,
                                             'gamma': gamma, 'u_0': u_0})

    store = GPprb.data_store

    if GPprb.early_stopping['enabled']:
        GPprb.store_eval_train_params(
            'val', {'X': X_val, 'y': y_val, 'bvalues': bvalues_val})
        args_val = store['common'] | store['val']
    else:
        X_train = np.vstack((X_train, X_val))
        y_train = np.vstack((y_train, y_val))
        bvalues_train = np.vstack((bvalues_train, bvalues_val))
        args_val = None

    GPprb.store_eval_train_params(
        'train', {'X': X_train, 'y': y_train, 'bvalues': bvalues_train})

    args_train = store['common'] | store['train']
    args_test_MSE = {'X': X_test, 'y': y_test, 'indlen': 0, 'bvalues': bvalues_test,
                     'S': S, 'bnodes': bnodes, 'gamma': gamma, 'u_0': u_0}
    args_test_sols = args_test_MSE.copy()
    args_test_sols['return_best_sol'] = True

    GPprb.register_eval_funcs(train_fit=eval_fitness.remote, args_train=args_train,
                              val_fit=eval_fitness.remote,
                              val_MSE=eval_MSE_remote.remote,
                              args_val=args_val, test_MSE=eval_MSE,
                              args_test_MSE=args_test_MSE, test_sols=eval_MSE,
                              args_test_sols=args_test_sols)

    if GPprb.plot_best:
        GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val, y=y_val,
                               bvalues=bvalues_val, S=S, bnodes=bnodes,
                               gamma=gamma, u_0=u_0,
                               toolbox=GPprb.toolbox, triang=triang)

    # MULTIPROCESSING ------------------------------------------------------------------
    def ray_mapper(f, individuals, toolbox):
        # We are not duplicating global scope on workers so we need to use the toolbox
        # Transform the tree expression in a callable function
        runnables = [toolbox.compile(expr=ind) for ind in individuals]
        lenghts = [len(ind) for ind in individuals]
        fitnesses = ray.get([f(*args) for args in zip(runnables, lenghts)])
        return fitnesses

    GPprb.toolbox.register("map", ray_mapper, toolbox=GPprb.toolbox)
    # ----------------------------------------------------------------------------------

    start = time.perf_counter()
    # opt_string = "SquareF(InnP0(InvMulP0(u, InnP0(u, fk)), delP1(dP0(u))))"
    # opt_string = "Sub(InnP1(dP0(u), dP0(u)), MulF(2, InnP0(fk, u)))"
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPprb.run(print_log=True, seed=None,
              save_best_individual=True, save_train_fit_history=True,
              save_best_test_sols=True, X_test=X_test,
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
