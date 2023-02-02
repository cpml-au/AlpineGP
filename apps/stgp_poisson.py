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
import sys
import yaml

# choose whether to use GPU or CPU and precision
config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

# suppress warnings
warnings.filterwarnings('ignore')

# generate mesh and dataset
S, bnodes, triang = d.generate_complex("test3.msh")
num_nodes = S.num_nodes

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

# noise factor
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
u_0 = C.CochainP0(S, u_0_vec, type="jax")


class ObjFunctional:
    def __init__(self) -> None:
        pass

    def setEnergyFunc(self, func, individual):
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func
        self.individual = individual

    def evalObjFunc(self, vec_x, vec_y, vec_bvalues):
        penalty = 0.5*gamma*jnp.sum((vec_x[bnodes] - vec_bvalues)**2)
        c = C.CochainP0(S, vec_x, "jax")
        fk = C.CochainP0(S, vec_y, "jax")
        energy = self.energy_func(c, fk) + penalty
        return energy


def evalPoissonObj(individual, X, y, current_bvalues, return_best_sol=False):
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
        if return_best_sol:
            return x

        current_result = np.linalg.norm(x-X[i, :])**2

        if current_result > 100 or math.isnan(current_result):
            current_result = 100

        result += current_result

    result *= 1/(num_sources*num_samples_per_source)
    length_penalty = min([np.abs(len(individual) - i) for i in range(1, 41)])
    terminal_penalty = int("u0" not in str(individual) or "u1" in str(individual))
    result += length_penalty
    result += 100*terminal_penalty

    return result,


GPproblem = gps.GPSymbRegProblem(pset,
                                 NINDIVIDUALS,
                                 NGEN,
                                 CXPB,
                                 MUTPB,
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
    u = evalPoissonObj(ind, X=X_train, y=y_train,
                       current_bvalues=bvalues_train, return_best_sol=True)
    plt.figure(10)
    fig = plt.gcf()
    plt.tricontourf(triang, u, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    fig.canvas.draw()
    plt.pause(0.05)


# copy GPproblem (needed to have a separate object shared among the processed
# to be modified in the last run)
FinalGP = GPproblem
FinalGP.toolbox.register("evaluate", evalPoissonObj, X=np.vstack((X_train, X_val)),
                         y=np.vstack((y_train, y_val)), current_bvalues=np.vstack((bvalues_train, bvalues_val)))


def stgp_poisson(config=None):
    # initialize list of best individuals
    best_individuals = []

    early_stopping = {'enabled': True, 'max_overfit': 3}
    final_training = True

    plot_best = True

    # multiprocessing parameters
    n_jobs = 4
    n_splits = 20

    if config is not None:
        GPproblem.NINDIVIDUALS = config["gp"]["NINDIVIDUALS"]
        GPproblem.NGEN = config["gp"]["NGEN"]
        n_jobs = config["mp"]["n_jobs"]
        n_splits = config["mp"]["n_splits"]
        early_stopping = config["gp"]["early_stopping"]
        GPproblem.parsimony_pressure = config["gp"]["parsimony_pressure"]
        FinalGP.parsimony_pressure = config["gp"]["parsimony_pressure"]
        final_training = config["gp"]["final_training"]
        mutate_fun = config["gp"]["mutate"]["fun"]
        mutate_kargs = eval(config["gp"]["mutate"]["kargs"])
        expr_mut_fun = config["gp"]["mutate"]["expr_mut"]
        expr_mut_kargs = eval(config["gp"]["mutate"]["expr_mut_kargs"])
        GPproblem.toolbox.register("mutate",
                                   eval(mutate_fun), **mutate_kargs)
        FinalGP.toolbox.register("mutate",
                                 eval(mutate_fun), **mutate_kargs)
        GPproblem.toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)
        FinalGP.toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)
        plot_best = config["plot"]["plot_best"]
        GPproblem.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        GPproblem.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    start = time.perf_counter()

    # add functions for fitness evaluation on training and validation sets to the
    # toolbox
    GPproblem.toolbox.register("evaluate_train",
                               evalPoissonObj,
                               X=X_train,
                               y=y_train,
                               current_bvalues=bvalues_train)
    GPproblem.toolbox.register("evaluate_val",
                               evalPoissonObj,
                               X=X_val,
                               y=y_val,
                               current_bvalues=bvalues_val)

    print("> MODEL TRAINING/SELECTION STARTED", flush=True)
    # train the model in the training set
    pool = mpire.WorkerPool(n_jobs=n_jobs)
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=True,
                  print_log=True,
                  plot_best=plot_best,
                  plot_best_func=plotSol,
                  seed=None,
                  n_splits=n_splits,
                  early_stopping=early_stopping)

    # Print best individual
    best = GPproblem.best
    print(f"The best individual is {str(best)}", flush=True)

    # evaluate score on the training and validation set
    print(f"The best score on the training set is {GPproblem.min_history[-1]}")
    print(f"The best score on the validation set is {GPproblem.min_valerr}")

    # FIXME: do I need it?
    best_individuals.append(best)
    # best_train_scores.append(score_train)
    # best_val_scores.append(score_train)

    print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

    if final_training:
        print("> FINAL TRAINING STARTED", flush=True)

        # update FinalGP.NGEN according to early stopping
        FinalGP.NGEN = GPproblem.NGEN

        # now we retrain the best model on the entire training set
        FinalGP.toolbox.register("map", pool.map)
        FinalGP.run(plot_history=True,
                    print_log=True,
                    plot_best=plot_best,
                    n_splits=n_splits,
                    seed=best_individuals)

        pool.terminate()
        best = tools.selBest(FinalGP.pop, k=1)[0]
        print(f"The best individual is {str(best)}", flush=True)

        score_train = FinalGP.min_history[-1]
        print(f"Best score on the training set = {score_train}")

    score_test = evalPoissonObj(best, X_test, y_test, bvalues_test)[0]
    print(f"Best score on the test set = {score_test}")

    print(f"Elapsed time: {round(time.perf_counter() - start, 2)}")

    # plot the tree of the best individual
    nodes, edges, labels = gp.graph(best)
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
    n_args = len(sys.argv)
    config = None
    if n_args > 1:
        param_file = sys.argv[1]
        print("Parameters file: ", param_file)
        with open(param_file) as file:
            config = yaml.safe_load(file)
            print(yaml.dump(config))
    stgp_poisson(config)
