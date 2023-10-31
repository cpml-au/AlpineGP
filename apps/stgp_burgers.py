from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
import matplotlib.pyplot as plt
from deap import gp, base, tools
from alpine.data.util import load_dataset
from alpine.data.burgers.burgers_dataset import data_path
from dctkit.mesh import util
from alpine.gp import gpsymbreg as gps
from dctkit import config
import dctkit as dt_
import pygmo as pg

import ray

import numpy as np
import math
import time
import sys
import yaml
from typing import Tuple, Callable
import numpy.typing as npt
import warnings
from jax import jit

residual_formulation = True

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()

# suppress all warnings
warnings.filterwarnings("ignore")


class Problem:
    def __init__(self, u: npt.NDArray, u_data_T: npt.NDArray, time_data: npt.NDArray,
                 dt: float, num_t_points: npt.NDArray, func: Callable,
                 S: SimplicialComplex):
        self.u = u
        self.u_data_T = u_data_T
        self.time_data = time_data
        self.dt = dt
        self.num_t_points = num_t_points
        self.func = func
        self.S = S
        self.func_coeffs = jit(self.set_func)

    def set_func(self, u_coeffs: npt.NDArray, epsilon: float):
        u = C.CochainD0(self.S, u_coeffs)
        return self.func(u, epsilon).coeffs

    def fitness(self, epsilon: float):
        total_err = 0
        for t in range(self.num_t_points - 1):
            self.u[1:-1, t+1] = self.u[1:-1, t] + self.dt * \
                self.func_coeffs(self.u[:, t], epsilon)[1:-1]
            if np.isnan(self.u[:, t+1]).any() or (np.abs(self.u[:, t+1]) > 1e5).any():
                total_err = np.nan
                break

        if math.isnan(total_err):
            total_err = 1e5

        else:
            # evaluate errors
            u_data = self.u_data_T.T
            errors = self.u[:, self.time_data] - u_data

            total_err = np.mean(np.linalg.norm(errors, axis=0)**2)

        return [total_err]

    def get_bounds(self):
        return ([0.00001], [1])


@ray.remote(num_cpus=2)
def tune_epsilon(func: Callable, epsilon: float, indlen: int, time_data: npt.NDArray,
                 u_data_T: npt.NDArray, bvalues: dict, S: SimplicialComplex,
                 num_t_points: float, num_x_points: float, dt: float,
                 u_0: C.CochainD0) -> Tuple[float, npt.NDArray]:
    # need to call config again before using JAX in energy evaluations to make sure
    # that the current worker has initialized JAX
    config()

    # initialize u setting initial and boundary conditions
    u = np.zeros((num_x_points-1, num_t_points), dtype=dt_.float_dtype)
    u[:, 0] = u_0.coeffs
    u[0, :] = bvalues['left']
    u[-1, :] = bvalues['right']

    prb = Problem(u, u_data_T, time_data, dt, num_t_points, func, S)
    algo = pg.algorithm(pg.de(gen=10))
    prob = pg.problem(prb)
    pop = pg.population(prob, size=200)
    pop = algo.evolve(pop)

    # extract epsilon
    epsilon = pop.champion_x[0]

    return epsilon


def eval_MSE_sol(func: Callable, epsilon: float, indlen: int, time_data: npt.NDArray,
                 u_data_T: npt.NDArray, bvalues: dict, S: SimplicialComplex,
                 num_t_points: float, num_x_points: float, dt: float,
                 u_0: C.CochainD0) -> Tuple[float, npt.NDArray]:

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # initialize u setting initial and boundary conditions
    u = np.zeros((num_x_points-1, num_t_points), dtype=dt_.float_dtype)
    u[:, 0] = u_0.coeffs
    u[0, :] = bvalues['left']
    u[-1, :] = bvalues['right']

    prb = Problem(u, u_data_T, time_data, dt, num_t_points, func, S)
    total_err = prb.fitness(epsilon)[0]

    # NOTE: since in our data we have u.T, we also store the transpose
    # of u as the best solution.
    best_sol = u[:, time_data].T

    return total_err, best_sol


@ray.remote(num_cpus=2)
def eval_best_sols(individual: Callable, epsilon: float, indlen: int,
                   time_data: npt.NDArray, u_data_T: npt.NDArray, bvalues: dict,
                   S: SimplicialComplex, num_t_points: float, num_x_points: float,
                   dt: float, u_0: C.CochainD0, penalty: dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(individual, epsilon, indlen, time_data,
                                u_data_T, bvalues, S, num_t_points, num_x_points, dt, u_0)

    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(individual: Callable, epsilon: float, indlen: int, time_data: npt.NDArray,
             u_data_T: npt.NDArray, bvalues: dict, S: SimplicialComplex,
             num_t_points: float, num_x_points: float, dt: float,
             u_0: C.CochainD0, penalty: dict) -> float:

    MSE, _ = eval_MSE_sol(individual, epsilon, indlen, time_data,
                          u_data_T, bvalues, S, num_t_points, num_x_points, dt, u_0)

    return MSE


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, epsilon: float, indlen: int,
                 time_data: npt.NDArray, u_data_T: npt.NDArray, bvalues: dict,
                 S: SimplicialComplex, num_t_points: float, num_x_points: float,
                 dt: float, u_0: C.CochainD0, penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, epsilon, indlen, time_data,
                                u_data_T, bvalues, S, num_t_points, num_x_points, dt, u_0)

    # penalty terms on length
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


# Plot best solution
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, bvalues: dict,
             S: SimplicialComplex, gamma: float, u_0: C.CochainD0,
             toolbox: base.Toolbox):

    indfun = toolbox.compile(expr=ind)
    dim = X.shape[0]

    _, u = eval_MSE_sol(indfun, indlen=0, X=X, bvalues=bvalues,
                        S=S, gamma=gamma, u_0=u_0)

    plt.figure(10, figsize=(10, 2))
    plt.clf()
    fig = plt.gcf()
    _, axes = plt.subplots(1, dim, num=10)
    for i in range(dim):
        axes[i].triplot(S.node_coords[:, 0], S.node_coords[:, 1],
                        triangles=S.S[2], color="#e5f5e0")
        axes[i].triplot(u[i][:, 0], u[i][:, 1],
                        triangles=S.S[2], color="#a1d99b")
        # axes[i].triplot(X[i, :, 0], X[i, :, 1],
        #                triangles=S.S[2], color="#4daf4a")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


def stgp_burgers(config_file, output_path=None):
    global residual_formulation

    # SPACE PARAMS
    L = 5
    L_norm = 1
    # spatial resolution
    dx = 0.05
    dx_norm = dx/L
    #  Number of spatial grid points
    num_x_points = int(L / dx)
    num_x_points_norm = num_x_points

    # vector containing spatial points
    x = np.linspace(0, L, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # initial velocity
    u_0 = 2 * np.exp(-2 * (x_circ - 0.5 * L)**2)
    umax = np.max(u_0)

    # TIME PARAMS
    T = 2
    T_norm = T*umax/L
    # temporal resolution
    dt = 0.01
    dt_norm = dt*umax/L
    # number of temporal grid points
    num_t_points_norm = int(T_norm / dt_norm)

    # generate mesh
    mesh, _ = util.generate_line_mesh(num_x_points_norm, L_norm)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_PDP_weights()

    # load data
    time_train, time_val, time_test, u_train_T, u_val_T, u_test_T = load_dataset(
        data_path, "npy")

    # initial condition
    u_0 = C.CochainD0(S, u_0/umax)

    # boundary conditions
    nodes_BC = {'left': np.zeros(num_t_points_norm),
                'right': np.zeros(num_t_points_norm)}

    residual_formulation = config_file["gp"]["residual_formulation"]
    use_ADF = config_file["gp"]["ADF"]["use_ADF"]

    # define primitive set and add primitives and terminals
    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, float], C.CochainD0)
        # add constants
        pset.addTerminal(0.5, float, name="1/2")
        pset.addTerminal(-0.5, float, name="-1/2")
        pset.addTerminal(-1., float, name="-1.")
        pset.addTerminal(2., float, name="2.")
        pset.addTerminal(-2., float, name="-2.")
        # pset.addTerminal(dx, float, name="dx")
        # pset.addTerminal(dt, float, name="dt")
        # pset.addTerminal(10., float, name="10.")
        # pset.addTerminal(0.1, float, name="0.1")
        # rename arguments
        pset.renameArguments(ARG0="u")
        pset.renameArguments(ARG1="eps")

        if use_ADF:
            ADF = gp.PrimitiveSetTyped("ADF", [C.CochainD0], C.CochainD1)
            pset.addADF(ADF)
            ADF.renameArguments(ARG0="u")
        else:
            ADF = None
    else:
        raise Exception("Only residual formulation available for this problem.")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file, ADF=ADF)

    penalty = config_file["gp"]["penalty"]

    GPprb.store_eval_common_params({'S': S,
                                    'penalty': penalty,
                                    'bvalues': nodes_BC,
                                    'dt': dt_norm,
                                    'num_t_points': num_t_points_norm,
                                    'num_x_points': num_x_points_norm,
                                    'u_0': u_0})

    params_names = ('time_data', 'u_data_T')
    datasets = {'train': [time_train, u_train_T],
                'val': [time_val, u_val_T],
                'test': [time_test, u_test_T]}
    GPprb.store_eval_dataset_params(params_names, datasets)

    GPprb.register_eval_funcs(fitness=eval_fitness.remote, error_metric=eval_MSE.remote,
                              test_sols=eval_best_sols.remote)

    # register custom functions
    GPprb.toolbox.register("evaluate_epsilon", tune_epsilon.remote,
                           time_data=time_train,
                           u_data_T=u_train_T,
                           bvalues=nodes_BC,
                           S=S,
                           dt=dt_norm,
                           num_t_points=num_t_points_norm,
                           num_x_points=num_x_points_norm,
                           u_0=u_0)

    # if GPprb.plot_best:
    #    GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val,
    #                           bvalues=bvalues_val, S=S, gamma=gamma, u_0=u_0,
    #                           toolbox=GPprb.toolbox)

    if use_ADF:
        GPprb.register_map([lambda ind: ind.epsilon, lambda x: len(x[0]) + len(x[1])])
    else:
        GPprb.register_map([lambda ind: ind.epsilon, len])

    def evaluate_epsilons(pop):
        if not hasattr(pop[0], "epsilon"):
            for ind in pop:
                ind.epsilon = 1.

        epsilons = GPprb.toolbox.map(GPprb.toolbox.evaluate_epsilon, pop)

        for ind, epsilon in zip(pop, epsilons):
            ind.epsilon = epsilon

    def print_epsilon(pop):
        best = tools.selBest(pop, k=1)[0]
        print("The best individual's epsilon is: ", best.epsilon)

    start = time.perf_counter()
    # from deap import creator
    # opt_string = "St1P1(cobP0(AddCP0(St1D1(flat_parD0(MFD0(SquareD0(u), -1/2))),
    # MFP0(St1D1(cobD0(u)),eps))))"
    # opt_string = "St1P1(cobP0(MFP0(SquareP0(St1D1(flat_upD0(u))),-1/2))))"
    # opt_string_MAIN = "St1P1(cobP0(MFP0(SquareP0(St1D1(ADF(u))),-1/2))))"
    # opt_string_ADF = "int_up(inter_up(u))"
    # opt_string_ADF = "flat_upD0(u)"
    # opt_individ_MAIN = creator.Tree.from_string(opt_string_MAIN, pset)
    # opt_individ_ADF = creator.Tree.from_string(opt_string_ADF, ADF)
    # opt_individ = creator.Individual([opt_individ_MAIN, opt_individ_ADF])
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPprb.run(print_log=True, seed=None,
              save_best_individual=True, save_train_fit_history=True,
              save_best_test_sols=True, X_test_param_name="u_data_T",
              output_path=output_path, preprocess_fun=evaluate_epsilons,
              callback_fun=print_epsilon)

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

    stgp_burgers(config_file, output_path)
