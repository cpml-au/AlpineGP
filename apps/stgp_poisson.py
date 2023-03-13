from dctkit.dec import cochain as C
import networkx as nx
import matplotlib.pyplot as plt
import deap
from deap import gp, tools
import dctkit
from alpine.models.poisson import pset, get_primitives_strings
from alpine.data import poisson_dataset as d
from alpine.gp import gpsymbreg as gps

import numpy as np
import warnings
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
import operator
import math
import mpire
import time
import sys
import yaml
import os

apps_path = os.path.dirname(os.path.realpath(__file__))


# generate mesh and dataset
S, bnodes, triang = d.generate_complex("test3.msh")
num_nodes = S.num_nodes
X_train, X_val, X_test, y_train, y_val, y_test = d.load_dataset()

# extract boundary values
bvalues_train = X_train[:, bnodes]
bvalues_val = X_val[:, bnodes]
bvalues_test = X_test[:, bnodes]

# set seed
seed = 42
deap.rng.seed(seed)
deap.np.random.seed(seed)

# choose precision and whether to use GPU or CPU
dctkit.config(dctkit.FloatDtype.float64, dctkit.IntDtype.int64,
              dctkit.Backend.jax, dctkit.Platform.cpu)


# suppress warnings
warnings.filterwarnings('ignore')


# list of types
types = [C.CochainP0, C.CochainP1, C.CochainP2,
         C.CochainD0, C.CochainD1, C.CochainD2, float]

# extract list of names of primitives
primitives_strings = get_primitives_strings(pset, types)

# penalty parameter for the Dirichlet bcs
gamma = 1000.

# initial guess for the solution
u_0_vec = 0.01*np.random.rand(num_nodes)
u_0 = C.CochainP0(S, u_0_vec)

# Objective function for minimization problem associated to the evaluation of each
# individual's fitness


class ObjFunction:
    def __init__(self) -> None:
        pass

    def set_energy_func(self, func, individual):
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func
        self.individual = individual

    def total_energy(self, vec_x, vec_y, vec_bvalues):
        penalty = 0.5*gamma*jnp.sum((vec_x[bnodes] - vec_bvalues)**2)
        c = C.CochainP0(S, vec_x)
        fk = C.CochainP0(S, vec_y)
        energy = self.energy_func(c, fk) + penalty
        return energy


def eval_MSE(individual: gp.PrimitiveTree, X: np.array, y: np.array, bvalues: dict, return_best_sol=False) -> float:
    """Evaluate total MSE over the dataset.

    Args:
        individual: individual to evaluate.
        X: samples of the dataset.
        y: targets of the dataset.
        bvalues: array containing the boundary values of the dataset functions.
        return_best_sol: True if we want the best solution (in this case the function
        returns it).

    Returns:
        total MSE over the dataset.
    """

    # transform the individual expression into a callable function
    energy_func = GPproblem.toolbox.compile(expr=individual)

    # create objective function and set its energy function
    obj = ObjFunction()
    obj.set_energy_func(energy_func, individual)

    # compute/compile jacobian of the objective wrt its first argument (vec_x)
    jac = jit(grad(obj.total_energy))

    total_err = 0.

    best_sols = []

    # TODO: parallelize using vmap once we can use jaxopt
    for i, vec_y in enumerate(y):
        # extract current bvalues
        vec_bvalues = bvalues[i, :]

        # minimize the objective
        x = minimize(fun=obj.total_energy, x0=u_0.coeffs,
                     args=(vec_y, vec_bvalues), method="L-BFGS-B", jac=jac).x
        if return_best_sol:
            best_sols.append(x)

        current_err = np.linalg.norm(x-X[i, :])**2

        if current_err > 100 or math.isnan(current_err):
            current_err = 100

        total_err += current_err

    if return_best_sol:
        return best_sols

    total_err *= 1/(X.shape[0])

    return total_err


def eval_fitness(individual: gp.PrimitiveTree, X: np.array, y: np.array, bvalues: dict, penalty: dict) -> (float, ):
    """Evaluate total fitness over the dataset.

    Args:
        individual: individual to evaluate.
        X: samples of the dataset.
        y: targets of the dataset.
        bvalues: np.array containing the boundary values of the dataset functions.
        penalty: dictionary containing the penalty method (regularization) and the
        penalty multiplier.

    Returns:
        total fitness over the dataset.
    """

    objval = 0.

    total_err = eval_MSE(individual, X, y, bvalues)

    if penalty["method"] == "primitive":
        # penalty terms on primitives
        indstr = str(individual)
        objval = total_err + penalty["reg_param"] * \
            max([indstr.count(string) for string in primitives_strings])
    elif penalty["method"] == "length":
        # penalty terms on length
        objval = total_err + penalty["reg_param"]*len(individual)
    else:
        # no penalty
        objval = total_err
    return objval,


# FIXME: maybe pass fitness function among parameters
GPproblem = gps.GPSymbRegProblem(pset)


# Plot best solution


def plot_sol(ind: gp.PrimitiveTree, X: np.array, y: np.array, bvalues: dict):
    u = eval_MSE(ind, X=X, y=y, bvalues=bvalues, return_best_sol=True)
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


def stgp_poisson(config_file):

    # set parameters from config file
    GPproblem.NINDIVIDUALS = config_file["gp"]["NINDIVIDUALS"]
    GPproblem.NGEN = config_file["gp"]["NGEN"]
    GPproblem.CXPB = config_file["gp"]["CXPB"]
    GPproblem.MUTPB = config_file["gp"]["MUTPB"]
    GPproblem.n_elitist = int(config_file["gp"]["frac_elitist"]*GPproblem.NINDIVIDUALS)
    min_ = config_file["gp"]["min_"]
    max_ = config_file["gp"]["max_"]
    early_stopping = config_file["gp"]["early_stopping"]
    GPproblem.parsimony_pressure = config_file["gp"]["parsimony_pressure"]
    penalty = config_file["gp"]["penalty"]

    GPproblem.tournsize = config_file["gp"]["select"]["tournsize"]
    GPproblem.stochastic_tournament = config_file["gp"]["select"]["stochastic_tournament"]

    mutate_fun = config_file["gp"]["mutate"]["fun"]
    mutate_kargs = eval(config_file["gp"]["mutate"]["kargs"])
    expr_mut_fun = config_file["gp"]["mutate"]["expr_mut"]
    expr_mut_kargs = eval(config_file["gp"]["mutate"]["expr_mut_kargs"])

    crossover_fun = config_file["gp"]["crossover"]["fun"]
    crossover_kargs = eval(config_file["gp"]["crossover"]["kargs"])
    GPproblem.toolbox.register("mate", eval(crossover_fun), **crossover_kargs)
    GPproblem.toolbox.register("mutate",
                               eval(mutate_fun), **mutate_kargs)
    GPproblem.toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)
    GPproblem.toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    GPproblem.toolbox.decorate(
        "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    plot_best = config_file["plot"]["plot_best"]
    plot_best_genealogy = config_file["plot"]["plot_best_genealogy"]

    n_jobs = config_file["mp"]["n_jobs"]
    n_splits = config_file["mp"]["n_splits"]
    start_method = config_file["mp"]["start_method"]

    GPproblem.toolbox.register("expr", gp.genHalfAndHalf,
                               pset=pset, min_=min_, max_=max_)
    GPproblem.toolbox.register("expr_pop",
                               gp.genHalfAndHalf,
                               pset=pset,
                               min_=min_,
                               max_=max_,
                               is_pop=True)
    GPproblem.toolbox.register("individual_pop", tools.initIterate,
                               GPproblem.createIndividual, GPproblem.toolbox.expr_pop)
    GPproblem.toolbox.register("population", tools.initRepeat,
                               list, GPproblem.toolbox.individual_pop)
    start = time.perf_counter()

    # add functions for fitness evaluation (value of the objective function) on training
    # set and MSE evaluation on validation set
    GPproblem.toolbox.register("evaluate_train",
                               eval_fitness,
                               X=X_train,
                               y=y_train,
                               bvalues=bvalues_train,
                               penalty=penalty)
    GPproblem.toolbox.register("evaluate_val_fit",
                               eval_fitness,
                               X=X_val,
                               y=y_val,
                               bvalues=bvalues_val,
                               penalty=penalty)
    GPproblem.toolbox.register("evaluate_val_MSE",
                               eval_MSE,
                               X=X_val,
                               y=y_val,
                               bvalues=bvalues_val)
    if plot_best:
        GPproblem.toolbox.register("plot_best_func", plot_sol,
                                   X=X_val, y=y_val, bvalues=bvalues_val)

    print("> MODEL TRAINING/SELECTION STARTED", flush=True)
    pool = mpire.WorkerPool(n_jobs=n_jobs, start_method=start_method)
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=True,
                  print_log=True,
                  plot_best=plot_best,
                  plot_best_genealogy=plot_best_genealogy,
                  seed=None,
                  n_splits=n_splits,
                  early_stopping=early_stopping)

    best = GPproblem.best
    print(f"The best individual is {str(best)}", flush=True)

    print(f"The best fitness on the training set is {GPproblem.train_fit_history[-1]}")
    print(f"The best fitness on the validation set is {GPproblem.min_valerr}")

    print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

    score_test = eval_MSE(best, X_test, y_test, bvalues_test)
    print(f"The best MSE on the test set is {score_test}")

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    # plot the tree of the best individual
    nodes, edges, labels = gp.graph(best)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="w")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.axis("off")
    plt.show()

    # save data for plots to disk
    np.save("train_fit_history.npy", GPproblem.train_fit_history)
    np.save("val_fit_history.npy", GPproblem.val_fit_history)

    best_sols = eval_MSE(best, X=X_test, y=y_test,
                         bvalues=bvalues_test, return_best_sol=True)

    for i, sol in enumerate(best_sols):
        np.save("best_sol_test_" + str(i) + ".npy", sol)
        np.save("true_sol_test_" + str(i) + ".npy", X_test[i])


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))
    stgp_poisson(config_file)
