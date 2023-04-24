import numpy as np
import jax.numpy as jnp
from deap import base, gp
from scipy import sparse
from scipy.linalg import block_diag
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh
from dctkit.dec import cochain as C
import dctkit as dt
from alpine.data.elastica_data import elastica_dataset as ed
from alpine.models.elastica import pset
from alpine.gp import gpsymbreg as gps
import math
import sys
import yaml
import time
import mpire
import networkx as nx
import matplotlib.pyplot as plt
from jax import grad, Array
from dctkit.math.opt import optctrl as oc
import numpy.typing as npt
from typing import Callable


# choose precision and whether to use GPU or CPU
dt.config(dt.FloatDtype.float64, dt.IntDtype.int64, dt.Backend.jax, dt.Platform.cpu)

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


def get_theta_0(x: list, y: list) -> list:
    """Routine to compute all the theta_0.

    Args:
        x (list): list in which each entry is the matrix of x-coordinates
        of the dataset.
        y (list): list in which each entry is the matrix of x-coordinates
        of the dataset.

    Returns:
        (list): list of all the theta_0

    """
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


def is_valid_energy(theta_0: npt.NDArray, theta: npt.NDArray, prb: oc.OptimizationProblem) -> bool:
    dim = len(theta_0)
    noise = 0.0001*np.ones(dim).astype(dt.float_dtype)
    theta_0_noise = theta_0 + noise
    theta_noise = prb.run(x0=theta_0_noise, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

    isnt_constant = np.allclose(theta, theta_noise, rtol=1e-6, atol=1e-6)
    return isnt_constant


class Objectives():
    def __init__(self, S: SimplicialComplex) -> None:
        self.S = S

    def set_energy_func(self, func: Callable, individual: gp.PrimitiveTree) -> None:
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func
        self.individual = individual

    # elastic energy including Dirichlet BC by elimination of the prescribed dofs
    def total_energy(self, theta_vec: npt.NDArray, FL2_EI0: float,
                     theta_in: npt.NDArray) -> Array:
        # extend theta on the boundary w.r.t boundary conditions
        theta_vec = jnp.insert(theta_vec, 0, theta_in)
        theta = C.CochainD0(self.S, theta_vec)
        FL2_EI0_coch = C.CochainD0(
            self.S, FL2_EI0*np.ones(self.S.num_nodes-1, dtype=dt.float_dtype))
        energy = self.energy_func(theta, FL2_EI0_coch)
        # jax.debug.print("{energy}", energy=energy)
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


def tune_EI0(individual: gp.PrimitiveTree, toolbox: base.Toolbox, FL2: float,
             EI0_guess: float, theta_guess: npt.NDArray, theta_true: npt.NDArray,
             S: SimplicialComplex) -> float:

    # transform the individual expression into a callable function
    energy_func = toolbox.compile(expr=individual)

    # number of unknowns angles
    dim = S.num_nodes-2

    obj = Objectives(S=S)
    obj.set_energy_func(energy_func, individual)

    theta_in = theta_true[0]

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


def eval_MSE(individual: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             toolbox: base.Toolbox, S: SimplicialComplex, theta_0_all: npt.NDArray,
             return_best_sol: bool = False) -> float:

    # transform the individual expression into a callable function
    energy_func = toolbox.compile(expr=individual)

    # number of unknowns angles
    dim = S.num_nodes-2

    total_err = 0.

    obj = Objectives(S=S)
    obj.set_energy_func(energy_func, individual)

    # init X_dim and best_theta
    X_dim = X.shape[0]
    best_theta = np.zeros((X_dim, S.num_nodes-1), dtype=dt.float_dtype)

    EI0 = individual.EI0

    if EI0 < 0:
        total_err = 10.
    else:
        for i, theta_true in enumerate(X):
            # extract prescribed value of theta at x = 0 from the dataset
            theta_in = theta_true[0]

            # get the right theta_0
            theta_0 = theta_0_all[i, :]
            # extract value of FL^2
            FL2 = y[i]
            # define value of the dimensionless parameter
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
                # print("prb.last_opt_result = ", prb.last_opt_result)
                # print("valid_energy = ", valid_energy)
                fval = math.nan

            # if fval is nan, the candidate can't be the solution
            if math.isnan(fval):
                total_err = 10.
                break

            # update the error: it is the sum of the error w.r.t. theta and
            # the error w.r.t. EI0
            total_err += fval

            # extend theta
            theta = np.insert(theta, 0, theta_in)
            # update best_theta
            best_theta[i, :] = theta

    if return_best_sol:
        return best_theta

    total_err *= 1/(X_dim)
    # round total_err to 5 decimal digits
    # NOTE: round doesn't work properly.
    # See https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
    total_err = float("{:.5f}".format(total_err))
    return 10*total_err


def eval_fitness(individual: gp.PrimitiveTree, X: np.array, y: np.array,
                 toolbox: base.Toolbox, S: SimplicialComplex, theta_0_all: np.array,
                 penalty: dict) -> tuple[float, ]:
    """Evaluate total fitness over the dataset.

    Args:
        individual: individual to evaluate.
        X: samples of the dataset.
        y: targets of the dataset.
        toolbox: toolbox used.
        S: simplicial complex.
        theta_0: initial theta.
        penalty: dictionary containing the penalty method (regularization) and the
        penalty multiplier.

    Returns:
        total fitness over the dataset.
    """

    objval = 0.

    total_err = eval_MSE(individual, X, y, toolbox, S, theta_0_all)

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


def plot_sol(ind: gp.PrimitiveTree, X: np.array, y: np.array, toolbox: base.Toolbox,
             S: SimplicialComplex, theta_0_all: np.array, transform: np.array, is_animated: bool = True) -> None:
    best_sol_all = eval_MSE(ind, X=X, y=y, toolbox=toolbox, S=S,
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
    if is_animated:
        plt.pause(0.1)
    else:
        plt.show()


def stgp_elastica(config_file):
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
    theta_0_all = get_theta_0(x_all, y_all)

    # define internal cochain
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0.
    internal_vec[-1] = 0.
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    # add it as a terminal
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")

    # initialize toolbox and creator
    createIndividual, toolbox = gps.creator_toolbox_config(
        config_file=config_file, pset=pset)

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

    plot_best = config_file["plot"]["plot_best"]
    plot_best_genealogy = config_file["plot"]["plot_best_genealogy"]

    n_jobs = config_file["mp"]["n_jobs"]
    n_splits = config_file["mp"]["n_splits"]
    start_method = config_file["mp"]["start_method"]

    start = time.perf_counter()

    # add functions for fitness evaluation (value of the objective function) on training
    # set and MSE evaluation on validation set
    toolbox.register("evaluate_EI0",
                     tune_EI0,
                     toolbox=toolbox,
                     FL2=y_train[0],
                     EI0_guess=1.,
                     theta_guess=theta_0_all[0][0, :],
                     theta_true=X_train[0, :],
                     S=S)
    toolbox.register("evaluate_train",
                     eval_fitness,
                     X=X_train,
                     y=y_train,
                     toolbox=toolbox,
                     S=S,
                     theta_0_all=theta_0_all[0],
                     penalty=penalty)
    toolbox.register("evaluate_val_fit",
                     eval_fitness,
                     X=X_val,
                     y=y_val,
                     toolbox=toolbox,
                     S=S,
                     theta_0_all=theta_0_all[1],
                     penalty=penalty)
    toolbox.register("evaluate_val_MSE",
                     eval_MSE,
                     X=X_val,
                     y=y_val,
                     toolbox=toolbox,
                     S=S,
                     theta_0_all=theta_0_all[1])

    if plot_best:
        toolbox.register("plot_best_func", plot_sol,
                         X=X_val, y=y_val, toolbox=toolbox,
                         S=S, theta_0_all=theta_0_all[1], transform=transform)

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

    # opt_string = "Add(SinF(SqrtF(Add(ExpF(CosF(InvF(SinF(2)))), Add(CosF(InnD1(CosD1(CochMulD1(SinD1(SqrtD1(St0(int_coch))), dD0(theta))), SubD1(dD0(FL2_EI0), St0(int_coch)))), SqrtF(SinF(2)))))), CosF(InnD0(AddD0(FL2_EI0, theta), ExpD0(InvMulD0(theta, 1/2)))))"
    # opt_individ = createIndividual.from_string(opt_string, pset)
    # seed = [opt_individ]

    print("> MODEL TRAINING/SELECTION STARTED", flush=True)
    pool = mpire.WorkerPool(n_jobs=n_jobs, start_method=start_method)
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=False,
                  print_log=True,
                  plot_best=plot_best,
                  plot_best_genealogy=plot_best_genealogy,
                  seed=None,
                  n_splits=n_splits,
                  early_stopping=early_stopping,
                  plot_freq=1,
                  is_elastica=True)

    best = GPproblem.best
    print(f"The best individual is {str(best)}", flush=True)

    print(f"The best fitness on the training set is {GPproblem.train_fit_history[-1]}")
    print(f"The best fitness on the validation set is {GPproblem.min_valerr}")

    print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

    # first tune EI0
    EI0 = eval_MSE(best, X_train, y_train, toolbox, S, theta_0_all[0])
    best.EI0 = EI0
    score_test = eval_MSE(best, X_test, y_test, toolbox, S, theta_0_all[2])
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

    best_sol = eval_MSE(best, X_test, y_test, toolbox, S, theta_0_all[2], True)
    np.save("best_sol_test_0.npy", best_sol)
    np.save("true_sol_test_0.npy", X_test)


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))
    stgp_elastica(config_file)
