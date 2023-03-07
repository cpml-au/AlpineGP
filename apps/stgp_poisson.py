from dctkit.dec import cochain as C
import networkx as nx
import matplotlib.pyplot as plt
import jax.config as config
import deap
from deap import gp
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

seed = 42
deap.rng.seed(seed)
deap.np.random.seed(seed)

# choose precision and whether to use GPU or CPU
dctkit.config(dctkit.FloatDtype.float64, dctkit.IntDtype.int64,
              dctkit.Backend.jax, dctkit.Platform.cpu)


# suppress warnings
warnings.filterwarnings('ignore')

# generate mesh and dataset
S, bnodes, triang = d.generate_complex("test3.msh")
num_nodes = S.num_nodes

# list of types
types = [C.CochainP0, C.CochainP1, C.CochainP2,
         C.CochainD0, C.CochainD1, C.CochainD2, float]

# extract list of names of primitives
primitives_strings = get_primitives_strings(pset, types)

# set default GP parameters
NINDIVIDUALS = 10
NGEN = 20
CXPB = 0.5
MUTPB = 0.1
frac_elitist = 0.1


# number of different source term functions to use to generate the dataset
num_sources = 3
# number of cases for each source term function
num_samples_per_source = 4
# whether to use validation dataset
use_validation = True

# noise
noise = 0.*np.random.rand(num_nodes)

data_X, data_y = d.generate_dataset(S, num_samples_per_source, num_sources, noise)
X, y = d.split_dataset(data_X, data_y, 0.25, 0.25, use_validation)

if use_validation:
    X_train, X_val, X_test = X
    y_train, y_val,  y_test = y
    # extract boundary values for the test set
    bvalues_test = X_test[:, bnodes]
else:
    X_train = X
    y_train = y

# extract bvalues_train
bvalues_train = X_train[:, bnodes]
bvalues_val = X_val[:, bnodes]

gamma = 1000.

# initial guess for the solution
u_0_vec = 0.01*np.random.rand(num_nodes)
u_0 = C.CochainP0(S, u_0_vec)


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


def eval_MSE(individual, X, y, bvalues, return_best_sol=False):
    """Evaluate total MSE over the dataset.

    Args:
        individual (gp.PrimitiveTree): individual to evaluate.
        X (np.array): samples of the dataset.
        y (np.array): targets of the dataset.
        bvalues (np.array): np.array containing the boundary values of the dataset functions.
        return_best_sol (bool): True if we want the best solution (in this case the function returns it),
        otherwise it is False.

    Returns:
        (float): total MSE over the dataset.
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

    total_err *= 1/(num_sources*num_samples_per_source)

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

    if penalty["method"] == "terminal":
        # penalty terms on terminals
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
GPproblem = gps.GPSymbRegProblem(pset,
                                 NINDIVIDUALS,
                                 NGEN,
                                 CXPB,
                                 MUTPB,
                                 stochastic_tournament={
                                     'enabled': True, 'prob': [0.7, 0.3]},
                                 tournsize=2,
                                 frac_elitist=frac_elitist)


# Register fitness function, selection and mutate operators
GPproblem.toolbox.register("mate", gp.cxOnePoint)
GPproblem.toolbox.register("expr_mut", gp.genGrow, min_=1, max_=3)
GPproblem.toolbox.register("mutate",
                           gp.mutUniform,
                           expr=GPproblem.toolbox.expr_mut,
                           pset=pset)

# Bloat control
GPproblem.toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
GPproblem.toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Plot best solution


def plotSol(ind):
    u = eval_MSE(ind, X=X_test, y=y_test,
                 bvalues=bvalues_test, return_best_sol=True)
    plt.figure(10, figsize=(8, 4))
    fig = plt.gcf()
    _, axes = plt.subplots(2, 3, num=10)
    for i in range(0, 3):
        axes[0, i].tricontourf(triang, u[i], cmap='RdBu', levels=20)
        pltobj = axes[1, i].tricontourf(triang, y_test[i], cmap='RdBu', levels=20)
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_box_aspect(1)
    plt.colorbar(pltobj, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


def stgp_poisson(config=None):

    # default parameters
    early_stopping = {'enabled': True, 'max_overfit': 3}
    plot_best = True
    plot_best_genealogy = False
    n_jobs = 4
    n_splits = 20
    start_method = "fork"
    penalty = {'method': "terminal", 'reg_param': 0.1}

    # set parameters from config file
    if config is not None:
        GPproblem.NINDIVIDUALS = config["gp"]["NINDIVIDUALS"]
        GPproblem.NGEN = config["gp"]["NGEN"]
        n_jobs = config["mp"]["n_jobs"]
        n_splits = config["mp"]["n_splits"]
        start_method = config["mp"]["start_method"]
        early_stopping = config["gp"]["early_stopping"]
        min_ = config["gp"]["min_"]
        max_ = config["gp"]["max_"]
        penalty = config["gp"]["penalty"]
        GPproblem.parsimony_pressure = config["gp"]["parsimony_pressure"]
        GPproblem.tournsize = config["gp"]["select"]["tournsize"]
        GPproblem.stochastic_tournament = config["gp"]["select"]["stochastic_tournament"]
        mutate_fun = config["gp"]["mutate"]["fun"]
        mutate_kargs = eval(config["gp"]["mutate"]["kargs"])
        crossover_fun = config["gp"]["crossover"]["fun"]
        crossover_kargs = eval(config["gp"]["crossover"]["kargs"])
        expr_mut_fun = config["gp"]["mutate"]["expr_mut"]
        expr_mut_kargs = eval(config["gp"]["mutate"]["expr_mut_kargs"])
        num_sources = config["dataset"]["num_sources"]
        num_samples_per_source = config["dataset"]["num_samples_per_source"]
        noise_param = config["dataset"]["noise_param"]
        GPproblem.toolbox.register("mate", eval(crossover_fun), **crossover_kargs)
        GPproblem.toolbox.register("mutate",
                                   eval(mutate_fun), **mutate_kargs)
        GPproblem.toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)
        GPproblem.toolbox.register("expr", gp.genHalfAndHalf,
                                   pset=pset, min_=min_, max_=max_)
        plot_best = config["plot"]["plot_best"]
        plot_best_genealogy = config["plot"]["plot_best_genealogy"]
        GPproblem.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        GPproblem.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    start = time.perf_counter()

    # load dataset
    noise = noise_param*np.random.rand(num_nodes)
    data_X, data_y = d.generate_dataset(S, num_samples_per_source, num_sources, noise)
    X, y = d.split_dataset(data_X, data_y, 0.25, 0.25, use_validation)

    X_train, X_val, X_test = X
    y_train, y_val,  y_test = y
    # extract boundary values for the test set
    bvalues_test = X_test[:, bnodes]
    # extract bvalues_train
    bvalues_train = X_train[:, bnodes]
    bvalues_val = X_val[:, bnodes]

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

    print("> MODEL TRAINING/SELECTION STARTED", flush=True)
    pool = mpire.WorkerPool(n_jobs=n_jobs, start_method=start_method)
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=True,
                  print_log=True,
                  plot_best=plot_best,
                  plot_best_func=plotSol,
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
        np.save("true_sol_test_" + str(i) + ".npy", y_test[i])


if __name__ == '__main__':
    n_args = len(sys.argv)
    config = None
    if n_args > 1:
        param_file = sys.argv[1]
        print("Parameters file: ", param_file)
        with open(param_file) as file:
            config = yaml.safe_load(file)
            print(yaml.dump(config))
    stgp_poisson(config)
