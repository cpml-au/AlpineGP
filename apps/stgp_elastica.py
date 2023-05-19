import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, Tuple
import jax.numpy as jnp
from jax import grad, Array
from deap import base, gp, tools
from scipy import sparse
from scipy.linalg import block_diag
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh
from dctkit.dec import cochain as C
from dctkit import config, FloatDtype, IntDtype, Backend, Platform
from dctkit.math.opt import optctrl as oc
import dctkit as dt
from alpine.data.elastica_data import elastica_dataset as ed
from alpine.models.elastica import pset
from alpine.gp import gpsymbreg as gps
import matplotlib.pyplot as plt
import math
import sys
import os
from os.path import join
import yaml
import time
# import networkx as nx
import ray
from ray.util.multiprocessing import Pool

apps_path = os.path.dirname(os.path.realpath(__file__))

# reducing the number of threads launched by eval_fitness
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config(FloatDtype.float64, IntDtype.int64, Backend.jax, Platform.cpu)

# list of types
types = [C.CochainP0, C.CochainP1, C.CochainD0, C.CochainD1, float]

# extract list of names of primitives
primitives_strings = gps.get_primitives_strings(pset, types)


def get_coords(X: tuple, transform: np.array) -> list:
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
    x_all = []
    y_all = []
    h = 1/X[0].shape[1]
    for i in range(len(X)):
        theta = X[i]
        dim = theta.shape[0]

        # compute cos and sin theta
        cos_theta = h*jnp.cos(theta)
        sin_theta = h*jnp.sin(theta)
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
        x_i = jnp.linalg.solve(T, b_x)
        y_i = jnp.linalg.solve(T, b_y)
        # reshape again to have a matrix
        x_i = x_i.reshape((theta.shape[0], theta.shape[1]+1))
        y_i = y_i.reshape((theta.shape[0], theta.shape[1]+1))
        # update the list
        x_all.append(x_i)
        y_all.append(y_i)
    return x_all, y_all


def theta_guesses(x: list, y: list) -> list:
    theta_0_all = []
    for i in range(3):
        x_current = x[i]
        y_current = y[i]
        theta_0_init = np.ones(
            (x_current.shape[0], x_current.shape[1]-2), dtype=dt.float_dtype)
        const_angles = np.arctan((y_current[:, -1] - y_current[:, 1]
                                  )/(x_current[:, -1] - x_current[:, 1]))
        theta_0_current = np.diag(const_angles) @ theta_0_init
        theta_0_all.append(theta_0_current)
    return theta_0_all


def is_valid_energy(theta_0: npt.NDArray, theta: npt.NDArray,
                    prb: oc.OptimizationProblem) -> bool:
    dim = len(theta_0)
    noise = 0.0001*np.ones(dim).astype(dt.float_dtype)
    theta_0_noise = theta_0 + noise
    theta_noise = prb.run(x0=theta_0_noise, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)
    isnt_constant = np.allclose(theta, theta_noise, rtol=1e-6, atol=1e-6)
    return isnt_constant


class Objectives():
    def __init__(self, S: SimplicialComplex) -> None:
        self.S = S

    def set_energy_func(self, func: Callable) -> None:
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func

    # elastic energy including Dirichlet BC by elimination of the prescribed dofs
    def total_energy(self, theta_vec: npt.NDArray, FL2_EI0: float,
                     theta_in: npt.NDArray) -> Array:
        # extend theta on the boundary w.r.t boundary conditions
        theta_vec = jnp.insert(theta_vec, 0, theta_in)
        theta = C.CochainD0(self.S, theta_vec)
        FL2_EI0_coch = C.CochainD0(
            self.S, FL2_EI0*jnp.ones(self.S.num_nodes-1, dtype=dt.float_dtype))
        energy = self.energy_func(theta, FL2_EI0_coch)
        return energy

    # state function: stationarity conditions for the total energy
    def total_energy_grad(self, x: npt.NDArray, theta_in: float) -> Array:
        theta = x[:-1]
        FL2_EI0 = x[-1]
        return grad(self.total_energy)(theta, FL2_EI0, theta_in)

    # objective function for the parameter EI0 identification problem
    def MSE_theta(self, x: npt.NDArray, theta_true: npt.NDArray) -> Array:
        theta = x[:-1]
        theta = jnp.insert(theta, 0, theta_true[0])
        return jnp.sum(jnp.square(theta-theta_true))


@ray.remote(num_cpus=2)
def tune_EI0(energy_func: Callable, EI0: float, indlen: int, FL2: float,
             EI0_guess: float, theta_guess: npt.NDArray,
             theta_true: npt.NDArray, S: SimplicialComplex) -> float:

    # number of unknowns angles
    dim = S.num_nodes-2

    obj = Objectives(S=S)
    obj.set_energy_func(energy_func)

    # prescribed angle at x=0
    theta_in = theta_true[0]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config(FloatDtype.float64, IntDtype.int64, Backend.jax, Platform.cpu)

    # run parameter identification on the first sample of the training set
    # set extra args for bilevel program
    constraint_args = {'theta_in': theta_in}
    obj_args = {'theta_true': theta_true}

    prb = oc.OptimalControlProblem(objfun=obj.MSE_theta,
                                   statefun=obj.total_energy_grad,
                                   state_dim=dim,
                                   nparams=S.num_nodes-1,
                                   constraint_args=constraint_args,
                                   obj_args=obj_args)

    def get_bounds():
        lb = -100*np.ones(dim+1, dt.float_dtype)
        ub = 100*np.ones(dim+1, dt.float_dtype)
        lb[-1] = -100
        ub[-1] = -1e-3
        return (lb, ub)

    prb.get_bounds = get_bounds

    FL2_EI0 = FL2/EI0_guess

    x0 = np.append(theta_guess, FL2_EI0)

    x = prb.run(x0=x0, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

    # theta = x[:-1]
    FL2_EI0 = x[-1]

    EI0 = FL2/FL2_EI0

    # if optimization failed, set negative EI0
    if not (prb.last_opt_result == 1 or prb.last_opt_result == 3):
        EI0 = -1

    return EI0


def eval_MSE(energy_func: Callable, EI0: float, indlen: int, X: npt.NDArray,
             y: npt.NDArray, S: SimplicialComplex, theta_0_all: npt.NDArray,
             return_best_sol: bool = False) -> float:

    # number of unknown angles
    dim = S.num_nodes-2

    total_err = 0.

    obj = Objectives(S=S)
    obj.set_energy_func(energy_func)

    # init X_dim and best_theta
    X_dim = X.shape[0]
    best_theta = np.zeros((X_dim, S.num_nodes-1), dtype=dt.float_dtype)

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config(FloatDtype.float64, IntDtype.int64, Backend.jax, Platform.cpu)

    if EI0 < 0:
        total_err = 40.
    else:
        for i, theta_true in enumerate(X):
            # extract prescribed value of theta at x = 0 from the dataset
            theta_in = theta_true[0]

            theta_0 = theta_0_all[i, :]

            FL2 = y[i]
            FL2_EI0 = FL2/EI0

            prb = oc.OptimizationProblem(
                dim=dim, state_dim=dim, objfun=obj.total_energy)
            args = {'FL2_EI0': FL2_EI0, 'theta_in': theta_in}
            prb.set_obj_args(args)
            theta = prb.run(x0=theta_0, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

            # check whether the energy is "admissible" (i.e. exclude constant energies
            # and energies with minima that are too sensitive to the initial guess)
            valid_energy = is_valid_energy(theta_0=theta_0, theta=theta, prb=prb)

            if (prb.last_opt_result == 1 or prb.last_opt_result == 3) and valid_energy:
                x = np.append(theta, FL2_EI0)
                fval = obj.MSE_theta(x, theta_true)
            else:
                fval = math.nan

            if math.isnan(fval):
                total_err = 40.
                break

            total_err += fval

            # extend theta
            theta = np.insert(theta, 0, theta_in)

            # update best_theta
            best_theta[i, :] = theta

    if return_best_sol:
        return best_theta

    total_err *= 1/(X_dim)

    # round total_err to 5 decimal digits
    total_err = float("{:.5f}".format(total_err))

    return 10*total_err


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, EI0: float, indlen: int, X: npt.NDArray,
                 y: npt.NDArray, S: SimplicialComplex, theta_0_all: npt.NDArray,
                 penalty: Dict, return_MSE=False) -> Tuple[float, ]:

    objval = 0.

    total_err = eval_MSE(individual, EI0, indlen, X, y, S, theta_0_all)

    # if penalty["method"] == "primitive":
    #     # penalty terms on primitives
    #     indstr = str(individual)
    #     objval = total_err + penalty["reg_param"] * \
    #         max([indstr.count(string) for string in primitives_strings])
    if penalty["method"] == "length":
        # penalty terms on length
        objval = total_err + penalty["reg_param"]*indlen
    else:
        # no penalty
        objval = total_err

    return objval,


def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             toolbox: base.Toolbox, S: SimplicialComplex, theta_0_all: npt.NDArray,
             transform: npt.NDArray) -> None:

    # the refs of these objects are not automatically converted to objects
    # (because we are not calling plot_sol via .remote())
    X = ray.get(X)
    S = ray.get(S)
    indfun = toolbox.compile(expr=ind)

    best_sol_all = eval_MSE(indfun, ind.EI0, indlen=0, X=X, y=y,  S=S,
                            theta_0_all=theta_0_all, return_best_sol=True)

    plt.figure(1, figsize=(10, 4))
    dim = X.shape[0]
    fig = plt.gcf()
    _, axes = plt.subplots(1, dim, num=1)
    # get the x,y coordinates LIST of the best and of the true
    x_curr, y_curr = get_coords((best_sol_all,), transform)
    x_true, y_true = get_coords((X,), transform)
    for i in range(dim):
        # plot the results
        axes[i].plot(x_true[0][i, :], y_true[0][i, :], 'r')
        axes[i].plot(x_curr[0][i, :], y_curr[0][i, :], 'b')

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.1)


def stgp_elastica(config_file, output_path=None):
    X_train, X_val, X_test, y_train, y_val, y_test = ed.load_dataset()

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

    # get (x,y) coordinates for the dataset
    X = X_train, X_val, X_test
    x_all, y_all = get_coords(X, transform)

    # get all theta_0
    theta_0_all = theta_guesses(x_all, y_all)

    # define internal cochain
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0.
    internal_vec[-1] = 0.
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    # add it as a terminal
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")

    # set parameters from config file
    GPproblem_settings, GPproblem_run, GPproblem_extra = gps.load_config_data(
        config_file_data=config_file, pset=pset)
    toolbox = GPproblem_settings['toolbox']
    penalty = GPproblem_extra['penalty']

    ray.init()

    # store shared objects refs
    S_ref = ray.put(S)
    penalty_ref = ray.put(penalty)
    X_train_ref = ray.put(X_train)
    y_train_ref = ray.put(y_train)
    X_val_ref = ray.put(X_val)
    y_val_ref = ray.put(y_val)

    if GPproblem_run['early_stopping']['enabled']:
        args_train = {'X': X_train_ref, 'y': y_train_ref, 'penalty': penalty_ref,
                      'S': S_ref, 'theta_0_all': theta_0_all[0], 'return_MSE': False}
        args_val_fit = {'X': X_val_ref, 'y': y_val_ref, 'penalty': penalty_ref,
                        'S': S_ref, 'theta_0_all': theta_0_all[1], 'return_MSE': False}
        args_val_MSE = {'X': X_val_ref, 'y': y_val_ref, 'penalty': penalty_ref,
                        'S': S_ref, 'theta_0_all': theta_0_all[1], 'return_MSE': True}
        # register functions for fitness/MSE evaluation on different datasets
        toolbox.register("evaluate_EI0", tune_EI0.remote, FL2=y_train[0],
                         EI0_guess=1., theta_guess=theta_0_all[0][0, :],
                         theta_true=X_train[0, :], S=S_ref)
        toolbox.register("evaluate_val_fit", eval_fitness.remote, **args_val_fit)
        toolbox.register("evaluate_val_MSE", eval_fitness.remote, **args_val_MSE)
    else:
        X_tr = np.vstack((X_train, X_val))
        y_tr = np.hstack((y_train, y_val))
        theta_0_all_new = np.vstack((theta_0_all[0], theta_0_all[1]))
        args_train = {'X': X_train_ref, 'y': y_train_ref, 'penalty': penalty_ref,
                      'S': S_ref, 'theta_0_all': theta_0_all_new[0],
                      'return_MSE': False}
        toolbox.register("evaluate_EI0", tune_EI0.remote, FL2=y_tr[0],
                         EI0_guess=1., theta_guess=theta_0_all_new[0, :],
                         theta_true=X_tr[0, :], S=S_ref)

    toolbox.register("evaluate_train", eval_fitness.remote, **args_train)

    if GPproblem_run['plot_best']:
        toolbox.register("plot_best_func", plot_sol, X=X_val_ref, y=y_val_ref,
                         toolbox=toolbox, S=S_ref, theta_0_all=theta_0_all[1],
                         transform=transform)

    GPproblem = gps.GPSymbRegProblem(pset=pset, **GPproblem_settings)

    # opt_string = ""
    # opt_individ = createIndividual.from_string(opt_string, pset)
    # seed = [opt_individ]

    # MULTIPROCESSING SETTINGS ---------------------------------------------------------
    pool = Pool()

    def ray_mapper(f, individuals, toolbox):
        # We are not duplicating global scope on workers so we need to use the toolbox
        # Transform the tree expression in a callable function
        runnables = [toolbox.compile(expr=ind) for ind in individuals]
        lenghts = [len(ind) for ind in individuals]
        EI0s = [ind.EI0 for ind in individuals]
        fitnesses = ray.get([f(*args) for args in zip(runnables, EI0s, lenghts)])
        return fitnesses

    GPproblem.toolbox.register("map", ray_mapper, toolbox=GPproblem.toolbox)
    # ----------------------------------------------------------------------------------

    def evaluate_EI0s(pop):
        if not hasattr(pop[0], "EI0"):
            for ind in pop:
                ind.EI0 = 1.

        EI0s = GPproblem.toolbox.map(GPproblem.toolbox.evaluate_EI0, pop)

        for ind, EI0 in zip(pop, EI0s):
            ind.EI0 = EI0

    def print_EI0(pop):
        best = tools.selBest(pop, k=1)[0]
        print("The best individual's EI0 is: ", best.EI0)

    start = time.perf_counter()

    GPproblem.run(plot_history=False, print_log=True, seed=None, **GPproblem_run,
                  preprocess_fun=evaluate_EI0s, callback_fun=print_EI0)

    # print stats on best individual at the end of the evolution
    best = GPproblem.best
    print(f"The best individual is {str(best)}", flush=True)
    print(f"The best fitness on the training set is {GPproblem.train_fit_history[-1]}")

    if GPproblem_run['early_stopping']['enabled']:
        print(f"The best fitness on the validation set is {GPproblem.min_valerr}")

    score_test = eval_MSE(GPproblem.toolbox.compile(expr=best), best.EI0,
                          len(str(best)), X_test, y_test, S=S,
                          theta_0_all=theta_0_all[2])

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

    stgp_elastica(config_file, output_path)
