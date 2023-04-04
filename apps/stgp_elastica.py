import numpy as np
import jax.numpy as jnp
from deap import base, gp, creator, tools
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh
from dctkit.math.opt import optctrl
from dctkit.dec import cochain as C
import dctkit as dt
from alpine.data.elastica_data import elastica_dataset as ed
from alpine.models.elastica import pset
from alpine.gp import gpsymbreg as gps
import math
import sys
import yaml
import operator
import time
import mpire
import networkx as nx
import matplotlib.pyplot as plt
import jax


# choose precision and whether to use GPU or CPU
dt.config(dt.FloatDtype.float64, dt.IntDtype.int64, dt.Backend.jax, dt.Platform.cpu)


class ObjElastica():
    def __init__(self, S: SimplicialComplex) -> None:
        self.S = S

    def set_energy_func(self, func: callable, individual: gp.PrimitiveTree):
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func
        self.individual = individual

    def total_energy(self, theta_vec: np.array, EI0: np.array, F: float, theta_in: np.array) -> float:
        """Energy found by the current individual.

        Args:
            theta_vec (np.array): values of theta (internally).
            EI0 (np.array): value of EI_0.
            theta_in (np.array): values of theta on the boundary.

        Returns:
            float: value of the energy.
        """
        # extend theta on the boundary w.r.t boundary conditions
        theta_vec = jnp.insert(theta_vec, 0, theta_in)
        theta = C.CochainD0(self.S, theta_vec)
        energy = self.energy_func(theta, EI0[0], F)
        return energy


def obj_fun_theta(theta_guess: np.array, EI_guess: np.array, theta_true: np.array) -> float:
    """Objective function for the bilevel problem (inverse problem).
    Args:
        theta_guess (np.array): candidate solution.
        EI_guess (np.array): candidate EI.
        theta_true (np.array): true solution.
    Returns:
        float: error between the candidate and the true theta
    """
    theta_guess = jnp.insert(theta_guess, 0, theta_true[0])
    return jnp.sum(jnp.square(theta_guess-theta_true))


def eval_MSE(individual: gp.PrimitiveTree, X: np.array, y: np.array, toolbox: base.Toolbox,
             S: SimplicialComplex, theta_0: np.array, return_best_sol: bool = False) -> float:
    # transform the individual expression into a callable function
    energy_func = toolbox.compile(expr=individual)

    # create objective function and set its energy function
    obj = ObjElastica(S)
    obj.set_energy_func(energy_func, individual)

    total_err = 0.
    best_theta = []
    best_EI0 = []

    EI0_0 = 1*np.ones(1, dtype=dt.float_dtype)
    if X.ndim == 1:
        iterate = enumerate(np.array([X]))
        y = np.array([y])
    else:
        iterate = enumerate(X)
    for i, theta_true in iterate:
        # extract boundary value in 0
        theta_in = theta_true[0]

        # extract value of F
        F = y[i, 1]

        # set extra args for bilevel program
        constraint_args = (F, theta_in)
        obj_args = (theta_true,)
        # set bilevel problem
        prb = optctrl.OptimalControlProblem(objfun=obj_fun_theta,
                                            state_en=obj.total_energy,
                                            state_dim=S.num_nodes-2,
                                            constraint_args=constraint_args,
                                            obj_args=obj_args)

        theta, EI0, fval = prb.run(theta_0, EI0_0, tol=1e-2)

        # extend theta
        theta = np.insert(theta, 0, theta_in)
        if return_best_sol:
            best_theta.append(theta)
            best_EI0.append(EI0)

        # if fval is nan, the candidate can't be the solution
        if math.isnan(fval):
            total_err = 100
            break

        # update the error: it is the sum of the error w.r.t. theta and
        # the error w.r.t. EI0
        current_err_theta = np.linalg.norm(theta - theta_true)
        current_err_EI0 = np.linalg.norm(EI0 - y[i, 0])
        total_err += current_err_theta + current_err_EI0

    if return_best_sol:
        return best_theta, best_EI0

    total_err *= 1/(X.shape[0])

    return total_err


def eval_fitness(individual: gp.PrimitiveTree, X: np.array, y: np.array, toolbox: base.Toolbox,
                 S: SimplicialComplex, theta_0: np.array, penalty: dict) -> (float,):
    """Evaluate total fitness over the dataset.

    Args:
        individual: individual to evaluate.
        X: samples of the dataset.
        y: targets of the dataset.
        toolbox: toolbox used.
        S:
        theta_0:
        penalty: dictionary containing the penalty method (regularization) and the
        penalty multiplier.

    Returns:
        total fitness over the dataset.
    """

    objval = 0.

    total_err = eval_MSE(individual, X, y, toolbox, S, theta_0)

    if penalty["method"] == "primitive":
        # penalty terms on primitives
        #indstr = str(individual)
        # objval = total_err + penalty["reg_param"] * \
        #    max([indstr.count(string) for string in primitives_strings])
        objval = total_err
    elif penalty["method"] == "length":
        # penalty terms on length
        objval = total_err + penalty["reg_param"]*len(individual)
    else:
        # no penalty
        objval = total_err
    return objval,


def stgp_elastica(config_file):
    X_train, X_val, X_test, y_train, y_val, y_test = ed.load_dataset()
    # get normalized simplicial complex
    S_1, x = generate_1_D_mesh(num_nodes=11, L=1)
    S = SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    # define internal cochain
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0
    internal_vec[-1] = 0
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    # add it as a terminal
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")

    # initial guess for the solution
    theta_0 = 0.01*np.random.rand(S.num_nodes-2)

    # initialize toolbox and creator
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual",
                   gp.PrimitiveTree,
                   fitness=creator.FitnessMin)
    createIndividual = creator.Individual
    # set parameters from config file
    NINDIVIDUALS = config_file["gp"]["NINDIVIDUALS"]
    NGEN = config_file["gp"]["NGEN"]
    CXPB = config_file["gp"]["CXPB"]
    MUTPB = config_file["gp"]["MUTPB"]
    frac_elitist = int(config_file["gp"]["frac_elitist"]*NINDIVIDUALS)
    min_ = config_file["gp"]["min_"]
    max_ = config_file["gp"]["max_"]
    overlapping_generation = config_file["gp"]["overlapping_generation"]
    early_stopping = config_file["gp"]["early_stopping"]
    parsimony_pressure = config_file["gp"]["parsimony_pressure"]
    penalty = config_file["gp"]["penalty"]

    tournsize = config_file["gp"]["select"]["tournsize"]
    stochastic_tournament = config_file["gp"]["select"]["stochastic_tournament"]

    expr_mut_fun = config_file["gp"]["mutate"]["expr_mut"]
    expr_mut_kargs = eval(config_file["gp"]["mutate"]["expr_mut_kargs"])

    toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)

    crossover_fun = config_file["gp"]["crossover"]["fun"]
    crossover_kargs = eval(config_file["gp"]["crossover"]["kargs"])

    mutate_fun = config_file["gp"]["mutate"]["fun"]
    mutate_kargs = eval(config_file["gp"]["mutate"]["kargs"])
    toolbox.register("mate", eval(crossover_fun), **crossover_kargs)
    toolbox.register("mutate",
                     eval(mutate_fun), **mutate_kargs)
    toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate(
        "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    plot_best = config_file["plot"]["plot_best"]
    plot_best_genealogy = config_file["plot"]["plot_best_genealogy"]

    n_jobs = config_file["mp"]["n_jobs"]
    n_splits = config_file["mp"]["n_splits"]
    start_method = config_file["mp"]["start_method"]

    toolbox.register("expr", gp.genHalfAndHalf,
                     pset=pset, min_=min_, max_=max_)
    toolbox.register("expr_pop",
                     gp.genHalfAndHalf,
                     pset=pset,
                     min_=min_,
                     max_=max_,
                     is_pop=True)
    toolbox.register("individual", tools.initIterate,
                     createIndividual, toolbox.expr)
    toolbox.register("individual_pop", tools.initIterate,
                     createIndividual, toolbox.expr_pop)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual_pop)
    toolbox.register("compile", gp.compile, pset=pset)
    start = time.perf_counter()

    # add functions for fitness evaluation (value of the objective function) on training
    # set and MSE evaluation on validation set
    toolbox.register("evaluate_train",
                     eval_fitness,
                     X=X_train,
                     y=y_train,
                     toolbox=toolbox,
                     S=S,
                     theta_0=theta_0,
                     penalty=penalty)
    toolbox.register("evaluate_val_fit",
                     eval_fitness,
                     X=X_val,
                     y=y_val,
                     toolbox=toolbox,
                     S=S,
                     theta_0=theta_0,
                     penalty=penalty)
    toolbox.register("evaluate_val_MSE",
                     eval_MSE,
                     X=X_val,
                     y=y_val,
                     toolbox=toolbox,
                     S=S,
                     theta_0=theta_0)
    '''
    if plot_best:
        toolbox.register("plot_best_func", plot_sol,
                         X=X_val, y=y_val, bvalues=bvalues_val,
                         S=S, bnodes=bnodes, gamma=gamma, u_0=u_0,
                         triang=triang, toolbox=toolbox)

    '''

    GPproblem = gps.GPSymbRegProblem(pset=pset,
                                     NINDIVIDUALS=NINDIVIDUALS,
                                     NGEN=NGEN,
                                     CXPB=CXPB,
                                     MUTPB=MUTPB,
                                     overlapping_generation=overlapping_generation,
                                     frac_elitist=frac_elitist,
                                     parsimony_pressure=parsimony_pressure,
                                     tournsize=tournsize,
                                     stochastic_tournament=stochastic_tournament,
                                     min_=min_,
                                     max_=max_,
                                     individualCreator=createIndividual,
                                     toolbox=toolbox)

    print("> MODEL TRAINING/SELECTION STARTED", flush=True)
    pool = mpire.WorkerPool(n_jobs=n_jobs, start_method=start_method)
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=False,
                  print_log=True,
                  seed=None,
                  n_splits=n_splits,
                  early_stopping=early_stopping)

    best = GPproblem.best
    print(f"The best individual is {str(best)}", flush=True)

    print(f"The best fitness on the training set is {GPproblem.train_fit_history[-1]}")
    print(f"The best fitness on the validation set is {GPproblem.min_valerr}")

    print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

    score_test = eval_MSE(best, X_test, y_test, toolbox, S, theta_0)
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

    # save graph in .txt file
    file = open("graph.txt", "w")
    a = file.write(str(best))
    file.close()

    # save data for plots to disk
    np.save("train_fit_history.npy", GPproblem.train_fit_history)
    np.save("val_fit_history.npy", GPproblem.val_fit_history)

    best_sols = eval_MSE(best, X_test, y_test, toolbox, S, theta_0, True)

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
    stgp_elastica(config_file)
