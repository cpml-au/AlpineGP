from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
# import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import tri
from deap import gp, base
from alpine.data.poisson import poisson_dataset as d
from alpine.gp import gpsymbreg as gps
from alpine.gp import primitives
from dctkit import config
import dctkit

from ray.util.multiprocessing import Pool
import ray

import numpy as np
import jax.numpy as jnp
import math
import time
import sys
import yaml
import os
from os.path import join
from typing import Tuple, Callable
import numpy.typing as npt

# reducing the number of threads launched by eval_fitness
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()

noise = d.load_noise()


def is_valid_energy(u: npt.NDArray, prb: oc.OptimizationProblem,
                    bnodes: npt.NDArray) -> bool:
    # perturb solution and check whether the gradient still vanishes
    # (i.e. constant energy)
    u_noise = u + noise*np.mean(u)
    u_noise[bnodes] = u[bnodes]
    grad_u_noise = jnp.linalg.norm(prb.gradient(u_noise))
    is_valid = grad_u_noise >= 1e-6
    return is_valid


def eval_MSE(residual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
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
        total_energy = C.inner_product(residual(c, fk), residual(c, fk)) + penalty
        return total_energy

    prb = oc.OptimizationProblem(
        dim=num_nodes, state_dim=num_nodes, objfun=total_energy)

    total_err = 0.

    best_sols = []

    # TODO: parallelize
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
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray, gamma: float,
                 u_0: npt.NDArray, penalty: dict, return_MSE=False) -> Tuple[float, ]:

    objval = 0.

    total_err = eval_MSE(individual, indlen, X, y, bvalues, S, bnodes, gamma, u_0)

    # if penalty["method"] == "primitive" and not return_MSE:
    #     # penalty terms on primitives
    #     indstr = str(individual)
    #     objval = total_err + penalty["reg_param"] * \
    #         max([indstr.count(string) for string in primitives_strings])
    if penalty["method"] == "length" and not return_MSE:
        # penalty terms on length
        objval = total_err + penalty["reg_param"]*indlen
    else:
        # no penalty
        objval = total_err

    return objval,


# Plot best solution
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: C.CochainP0, toolbox: base.Toolbox,
             triang: tri.Triangulation):

    # the refs of these objects are not automatically converted to objects
    # (because we are not calling plot_sol via .remote())
    bnodes = ray.get(bnodes)
    bvalues = ray.get(bvalues)
    gamma = ray.get(gamma)
    u_0 = ray.get(u_0)
    S = ray.get(S)
    indfun = toolbox.compile(expr=ind)

    u = eval_MSE(indfun, indlen=0, X=X, y=y, bvalues=bvalues, S=S,
                 bnodes=bnodes, gamma=gamma, u_0=u_0, return_best_sol=True)

    plt.figure(10, figsize=(8, 4))
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
    S, bnodes, triang = d.generate_complex(0.08)
    num_nodes = S.num_nodes
    X_train, X_val, X_test, y_train, y_val, y_test = d.load_dataset()

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
    pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], C.Cochain)
    pset.addTerminal(C.Cochain(S.num_nodes, True, S, np.ones(
        S.num_nodes, dtype=dctkit.float_dtype)), C.Cochain, name="ones")

    # set parameters from config file
    GPproblem_settings, GPproblem_run, GPproblem_extra = gps.load_config_data(
        config_file_data=config_file, pset=pset)
    toolbox = GPproblem_settings['toolbox']
    penalty = GPproblem_extra['penalty']
    # n_jobs = GPproblem_extra['n_jobs']

    primitives.addPrimitivesToPset(pset, GPproblem_settings['primitives'])

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1")
    pset.addTerminal(2., float, name="2")

    # rename arguments
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="fk")

    GPproblem_settings.pop('primitives')

    ray.init()

    # store shared objects refs
    S_ref = ray.put(S)
    penalty_ref = ray.put(penalty)
    bnodes_ref = ray.put(bnodes)
    gamma_ref = ray.put(gamma)
    u_0_ref = ray.put(u_0)
    X_train_ref = ray.put(X_train)
    y_train_ref = ray.put(y_train)
    bvalues_train_ref = ray.put(bvalues_train)
    bvalues_val_ref = ray.put(bvalues_val)

    # set arguments for evaluate functions
    if GPproblem_run['early_stopping']['enabled']:
        args_train = {'X': X_train_ref, 'y': y_train_ref, 'bvalues': bvalues_train_ref,
                      'penalty': penalty_ref, 'S': S_ref, 'bnodes': bnodes_ref,
                      'gamma': gamma_ref, 'u_0': u_0_ref, 'return_MSE': False}
        args_val_fit = {'X': X_val, 'y': y_val, 'bvalues': bvalues_val_ref,
                        'penalty': penalty_ref, 'S': S_ref, 'bnodes': bnodes_ref,
                        'gamma': gamma_ref, 'u_0': u_0_ref, 'return_MSE': False}
        args_val_MSE = {'X': X_val, 'y': y_val, 'bvalues': bvalues_val_ref,
                        'penalty': penalty_ref, 'S': S_ref, 'bnodes': bnodes_ref,
                        'gamma': gamma_ref, 'u_0': u_0_ref, 'return_MSE': True}
        # register functions for fitness/MSE evaluation on different datasets
        toolbox.register("evaluate_val_fit", eval_fitness.remote, **args_val_fit)
        toolbox.register("evaluate_val_MSE", eval_fitness.remote, **args_val_MSE)
    else:
        X_tr = np.vstack((X_train, X_val))
        y_tr = np.vstack((y_train, y_val))
        X_tr_ref = ray.put(X_tr)
        y_tr_ref = ray.put(y_tr)
        bvalues_tr = np.vstack((bvalues_train, bvalues_val))
        bvalues_tr_ref = ray.put(bvalues_tr)
        args_train = {'X': X_tr_ref, 'y': y_tr_ref, 'bvalues': bvalues_tr_ref,
                      'penalty': penalty_ref, 'S': S_ref, 'bnodes': bnodes_ref,
                      'gamma': gamma_ref, 'u_0': u_0_ref, 'return_MSE': False}

    # register functions for fitness/MSE evaluation on different datasets
    toolbox.register("evaluate_train", eval_fitness.remote, **args_train)

    if GPproblem_run['plot_best']:
        toolbox.register("plot_best_func", plot_sol, X=X_val, y=y_val,
                         bvalues=bvalues_val_ref, S=S_ref, bnodes=bnodes_ref,
                         gamma=gamma_ref, u_0=u_0_ref, toolbox=toolbox, triang=triang)

    # create symbolic regression problem instance
    GPproblem = gps.GPSymbRegProblem(pset=pset, **GPproblem_settings)

    # MULTIPROCESSING SETTINGS ---------------------------------------------------------
    pool = Pool()

    def ray_mapper(f, individuals, toolbox):
        # We are not duplicating global scope on workers so we need to use the toolbox
        # Transform the tree expression in a callable function
        runnables = [toolbox.compile(expr=ind) for ind in individuals]
        lenghts = [len(ind) for ind in individuals]
        fitnesses = ray.get([f(*args) for args in zip(runnables, lenghts)])
        return fitnesses

    GPproblem.toolbox.register("map", ray_mapper, toolbox=GPproblem.toolbox)
    # ----------------------------------------------------------------------------------

    start = time.perf_counter()
    # opt_string = "SquareF(InnP0(InvMulP0(u, InnP0(u, fk)), delP1(dP0(u))))"
    # opt_string = "SquareF(InnP0(InvMulP0(u, SquareF(InnP0(u, fk))), delP1(dP0(u))))"
    # opt_string = "Inn1(dP0(AddP0(SinP0(SqrtP0(u)), LogP0(u))), dP0(u))"
    # opt_string = "Sub(InnP1(dP0(u), dP0(u)), MulF(2, InnP0(fk, u)))"
    # opt_string = "MulF(2, Inn0(fk, fk)))"
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPproblem.run(plot_history=False, print_log=True, seed=None, **GPproblem_run)

    best = GPproblem.best

    score_test = eval_MSE(GPproblem.toolbox.compile(expr=best), len(str(best)),
                          X_test, y_test, bvalues_test, S=S,
                          bnodes=bnodes, gamma=gamma, u_0=u_0)

    print(f"The best MSE on the test set is {score_test}")

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    pool.close()
    ray.shutdown()

    # plot the tree of the best individual
    # nodes, edges, labels = gp.graph(best)
    # graph = nx.Graph()
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    # pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    # plt.figure(figsize=(7, 7))
    # nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="w")
    # nx.draw_networkx_edges(graph, pos)
    # nx.draw_networkx_labels(graph, pos, labels)
    # plt.axis("off")
    # plt.show()

    # save string of best individual in .txt file
    file = open(join(output_path, "best_ind.txt"), "w")
    file.write(str(best))
    file.close()

    # save data for plots to disk
    np.save(join(output_path, "train_fit_history.npy"),
            GPproblem.train_fit_history)
    if GPproblem_run['early_stopping']['enabled']:
        np.save(join(output_path, "val_fit_history.npy"), GPproblem.val_fit_history)

    best_sols = eval_MSE(GPproblem.toolbox.compile(expr=best), len(str(best)), X=X_test,
                         y=y_test, bvalues=bvalues_test, S=S, bnodes=bnodes,
                         gamma=gamma, u_0=u_0, return_best_sol=True)

    for i, sol in enumerate(best_sols):
        np.save(join(output_path, "best_sol_test_" + str(i) + ".npy"), sol)
        np.save(join(output_path, "true_sol_test_" + str(i) + ".npy"), X_test[i])


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
