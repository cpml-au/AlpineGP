import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
import jax.config as config
from deap import gp, tools
from dctkit.dec import cochain as C
from alpine.models.poisson import pset
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

# choose whether to use GPU or CPU and precision
config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

# suppress warnings
warnings.filterwarnings('ignore')

# generate mesh and dataset
S, bnodes = d.generate_complex("test3.msh")
num_nodes = S.num_nodes

# set GP parameters
NINDIVIDUALS = 100
NGEN = 20
CXPB = 0.5
MUTPB = 0.1

# number of different source term functions to use to generate the dataset
num_sources = 3
# number of cases for each source term function
num_samples_per_source = 2
# whether to use validation dataset
use_validation = True

X, y = d.split_dataset(S, num_samples_per_source, num_sources, 0.25, use_validation)

if use_validation:
    X_train, X_test = X
    y_train, y_test = y
    # extract boundary values for the test set
    bvalues_test = X_test[:, bnodes]
    Xval, yval = d.split_dataset(S, num_samples_per_source,
                                 num_sources, 0.25, use_validation)
    X_t, X_val = Xval
    y_t, y_val = yval
else:
    X_train = X
    y_train = y

# extract bvalues_train
bvalues_train = X_train[:, bnodes]

gamma = 1000.

# initial guess for the solution
u_0_vec = 0.01*np.random.rand(num_nodes)
u_0 = C.CochainP0(S, u_0_vec, type="jax")


class ObjFunctional:
    def __init__(self) -> None:
        pass

    def setEnergyFunc(self, func, individual):
        """Set the energy function to be used for the computation of the objective function."""
        self.energy_func = func
        self.individual = individual

    def evalObjFunc(self, vec_x, vec_y, vec_bvalues):
        penalty = 0.5*gamma*jnp.sum((vec_x[bnodes] - vec_bvalues)**2)
        c = C.CochainP0(S, vec_x, "jax")
        fk = C.CochainP0(S, vec_y, "jax")
        energy = self.energy_func(c, fk) + penalty
        return energy


def evalPoisson(individual, X, y, current_bvalues):
    # discard individuals with too many terminals (avoid error during compilation)
    if len(individual) > 50:
        result = 1000
        return result,

    # transform the individual expression into a callable function
    energy_func = GPproblem.toolbox.compile(expr=individual)

    # create objective function and set its energy function
    obj = ObjFunctional()
    obj.setEnergyFunc(energy_func, individual)

    # compute/compile jacobian of the objective wrt its first argument (vec_x)
    jac = jit(grad(obj.evalObjFunc))

    result = 0

    # TODO: parallelize using vmap once we can use jaxopt
    for i, vec_y in enumerate(y):
        # extract current bvalues
        vec_bvalues = current_bvalues[i, :]

        # minimize the objective
        x = minimize(fun=obj.evalObjFunc, x0=u_0.coeffs,
                     args=(vec_y, vec_bvalues), method="L-BFGS-B", jac=jac).x
        current_result = np.linalg.norm(x-X[i, :])**2

        if current_result > 100 or math.isnan(current_result):
            current_result = 100

        result += current_result

    result = 1/(num_sources*num_samples_per_source)*result

    return result,


GPproblem = gps.GPSymbRegProblem(pset,
                                 NINDIVIDUALS,
                                 NGEN,
                                 CXPB,
                                 MUTPB)

# Register fitness function, selection and mutate operators
GPproblem.toolbox.register(
    "select", GPproblem.selElitistAndTournament, frac_elitist=0.1)
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

# copy GPproblem (needed to have a separate object shared among the processed to be modified in the last run)
FinalGP = GPproblem
FinalGP.toolbox.register("evaluate", evalPoisson, X=X_train,
                         y=y_train, current_bvalues=bvalues_train)


def stgp_poisson():
    # initialize list of best individuals and list of best scores
    best_individuals = []
    best_train_scores = []
    best_val_scores = []

    start = time.perf_counter()

    # define current bvalues datasets
    bvalues_train = X_t[:, bnodes]
    bvalues_val = X_val[:, bnodes]

    # update toolbox
    GPproblem.toolbox.register("evaluate",
                               evalPoisson,
                               X=X_t,
                               y=y_t,
                               current_bvalues=bvalues_train)

    print("> MODEL TRAINING/SELECTION STARTED", flush=True)
    # train the model in the training set
    pool = mpire.WorkerPool()
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=True,
                  print_log=True,
                  plot_best=True,
                  seed=None)

    # Print best individual
    best = tools.selBest(GPproblem.pop, k=1)
    print(f"The best individual is {str(best[0])}", flush=True)

    # evaluate score on the current training and validation set
    score_train = GPproblem.min_history[-1]
    score_val = evalPoisson(best[0], X_val, y_val, bvalues_val)[0]

    print(f"Best score on the training set = {score_train}")
    print(f"Best score on the validation set = {score_val}")

    # save best individual and best score on training and validation set
    best_individuals.append(best[0])

    # FIXME: do I need it?
    best_train_scores.append(score_train)
    best_val_scores.append(score_train)

    print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

    print("> FINAL TRAINING STARTED", flush=True)

    # now we retrain the best model on the entire training set
    # FinalGP.toolbox.register("evaluate", evalPoisson, X=X_train,
    #                          y=y_train, current_bvalues=bvalues_train)
    FinalGP.toolbox.register("map", pool.map)
    FinalGP.run(plot_history=True,
                print_log=True,
                plot_best=True,
                seed=best_individuals)

    pool.terminate()
    real_best = tools.selBest(FinalGP.pop, k=1)
    print(f"The best individual is {str(real_best[0])}", flush=True)

    score_train = FinalGP.min_history[-1]
    score_test = evalPoisson(real_best[0], X_test, y_test, bvalues_test)
    score_test = score_test[0]

    print(f"Best score on the training set = {score_train}")
    print(f"Best score on the test set = {score_test}")

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    # plot the tree of the best individual
    nodes, edges, labels = gp.graph(real_best[0])
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    # pos = graphviz_layout(graph, prog='dot')
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="w")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    stgp_poisson()
