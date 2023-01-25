import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
import jax.config as config
from deap import gp, tools
from dctkit.dec import cochain as C
from alpine.apps.poisson import pset
from alpine.data import poisson_dataset as d
from alpine.gp import gpsymbreg as gps

import numpy as np
import warnings
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
import operator
import math


config.update("jax_enable_x64", True)

# generate mesh and dataset
S, bnodes = d.generate_complex("test3.msh")
dim_0 = S.num_nodes
NINDIVIDUALS = 10
NGEN = 1
CXPB = 0.5
MUTPB = 0.1
num_data = 2
diff = 3
k = 2
is_valid = True
X, y, kf = d.split_dataset(S, num_data, diff, k, is_valid)

if is_valid:
    X_train, X_test = X
    y_train, y_test = y
    # extract bvalues_test
    bvalues_test = X_test[:, bnodes]
    # dataset = (X_train, y_train, X_test, y_test)
else:
    # dataset = (X, y)
    X_train = X
    y_train = y

# extract bvalues_train
bvalues_train = X_train[:, bnodes]

gamma = 1000.
u_0_vec = 0.01*np.random.rand(dim_0)
u_0 = C.CochainP0(S, u_0_vec, type="jax")


class ObjFunctional:
    def __init__(self) -> None:
        pass

    def setFunc(self, func, individual):
        self.energy_func = func
        self.individual = individual

    def evalEnergy(self, vec_x, vec_y, vec_bvalues):
        # Transform the tree expression in a callable function
        penalty = 0.5*gamma*jnp.sum((vec_x[bnodes] - vec_bvalues)**2)
        # jax.debug.print("{x}", x=penalty)
        c = C.CochainP0(S, vec_x, "jax")
        fk = C.CochainP0(S, vec_y, "jax")
        # jax.debug.print("{x}", x=jnp.linalg.norm(c.coeffs - f.coeffs))
        energy = self.energy_func(c, fk) + penalty
        # jax.debug.print("{x}", x=energy)
        return energy


# suppress warnings
warnings.filterwarnings('ignore')


def evalPoisson(individual, X, y, current_bvalues):
    # NOTE: we are introducing a BIAS...
    if len(individual) > 50:
        result = 1000
        # print(result)
        return result,

    energy_func = GPproblem.toolbox.compile(expr=individual)

    # the current objective function is the current energy
    obj = ObjFunctional()
    obj.setFunc(energy_func, individual)

    result = 0
    for i, vec_y in enumerate(y):
        # minimize the energy w.r.t. data
        jac = jit(grad(obj.evalEnergy))

        # extract current bvalues
        vec_bvalues = current_bvalues[i, :]

        x = minimize(fun=obj.evalEnergy, x0=u_0.coeffs,
                     args=(vec_y, vec_bvalues), method="BFGS", jac=jac).x
        current_result = np.linalg.norm(x-X[i, :])**2

        # to avoid strange numbers, if result is too distance from 0 or is nan we
        # assign to it a default big number
        if current_result > 100 or math.isnan(current_result):
            current_result = 100

        result += current_result

    result = 1/(diff*num_data)*result
    # length_factor = math.prod([1 - i/len(individual)
    # for i in range(0, 50)])
    # penalty_length = gamma*abs(length_factor)
    # result += penalty_length
    return result,


GPproblem = gps.GPSymbRegProblem(pset,
                                 NINDIVIDUALS,
                                 NGEN,
                                 CXPB,
                                 MUTPB,
                                 min_=1,
                                 max_=4)

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


FinalGP = GPproblem

# Set toolbox for FinalGP
FinalGP.toolbox.register("evaluate", evalPoisson, X=X_train,
                         y=y_train, current_bvalues=bvalues_train)


def test_stgp_poisson():
    # initialize list of best individuals and list of best scores
    best_individuals = []
    best_train_scores = []
    best_val_scores = []

    # extract dataset
    if kf != 0:
        # X_train, y_train, X_test, y_test = dataset

        # start learning
        for train_index, valid_index in kf.split(X_train, y_train):
            # divide the dataset in training and validation set
            X_t, X_val = X_train[train_index, :], X_train[valid_index, :]
            y_t, y_val = y_train[train_index, :], y_train[valid_index, :]

            # define current bvalues datasets
            current_bvalues_train = X_t[:, bnodes]
            current_bvalues_val = X_val[:, bnodes]

            # update toolbox
            GPproblem.toolbox.register("evaluate",
                                       evalPoisson,
                                       X=X_t,
                                       y=y_t,
                                       current_bvalues=current_bvalues_train)

            # train the model in the training set
            print("Starting the pool...")
            pool = multiprocessing.Pool()
            print("Multiprocessing completed")
            GPproblem.toolbox.register("map", pool.map)
            GPproblem.run(plot_history=True,
                          print_log=True,
                          plot_best=True,
                          seed=None)
            pool.close()
            print("Multiprocessing closed")

            # Print best individual
            best = tools.selBest(GPproblem.pop, k=1)
            print(f"The best individual in this fold is {str(best[0])}")

            # evaluate score on the current training and validation set
            score_train = GPproblem.min_history[-1]
            score_val = evalPoisson(best[0], X_val, y_val, current_bvalues_val)
            score_val = score_val[0]

            print(f"The best score on training set in this fold is {score_train}")
            print(f"The best score on validation set in this fold is {score_val}")

            # save best individual and best score on training and validation set
            best_individuals.append(best[0])

            # FIXME: do I need it?
            best_train_scores.append(score_train)
            best_val_scores.append(score_train)

            print("-FOLD COMPLETED-")

        print("-FINAL TRAINING STARTED-")

        # now we retrain all the k best models on the entire training set
        print("Starting the pool...")
        pool = multiprocessing.Pool()
        print("Multiprocessing completed")
        FinalGP.toolbox.register("map", pool.map)
        FinalGP.run(plot_history=True,
                    print_log=True,
                    plot_best=True,
                    seed=best_individuals)
        pool.close()
        print("Multiprocessing closed")
        real_best = tools.selBest(FinalGP.pop, k=1)

        score_train = FinalGP.min_history[-1]
        score_test = evalPoisson(real_best[0], X_test, y_test, bvalues_test)
        score_test = score_test[0]

        print(f"The best score on training set is {score_train}")
        print(f"The best score on test set is {score_test}")

    else:
        # X_train, y_train = dataset
        # now we retrain all the k best models on the entire training set
        print("Starting the pool...")
        pool = multiprocessing.Pool()
        print("Multiprocessing completed")
        FinalGP.toolbox.register("map", pool.map)
        FinalGP.run(plot_history=True,
                    print_log=True,
                    plot_best=True,
                    seed=None)
        pool.close()
        print("Multiprocessing closed")
        real_best = tools.selBest(FinalGP.pop, k=1)

    # plot the best solution
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
    test_stgp_poisson()
