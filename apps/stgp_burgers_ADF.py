from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
import matplotlib.pyplot as plt
from deap import gp, base
from alpine.data.util import load_dataset
from alpine.data.burgers.burgers_dataset import data_path
from dctkit.mesh import util
from alpine.gp import gpsymbreg as gps
from dctkit import config
import dctkit as dt_

import ray

import numpy as np
import math
import time
import sys
import yaml
from typing import Tuple, Callable
import numpy.typing as npt
import warnings

residual_formulation = True

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()

# suppress all warnings
warnings.filterwarnings("ignore")


def eval_MSE_sol(func: Callable, indlen: int, time_data: npt.NDArray, u_data_T: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, num_t_points: float,
                 num_x_points: float, dt: float, u_0: C.CochainD0) -> Tuple[float, npt.NDArray]:

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # initialize u setting initial and boundary conditions
    u = np.zeros((num_x_points-1, num_t_points), dtype=dt_.float_dtype)
    u[:, 0] = u_0.coeffs
    u[0, :] = bvalues['left']
    u[-1, :] = bvalues['right']

    total_err = 0.
    # main loop
    for t in range(num_t_points - 1):
        u_coch = C.CochainD0(S, u[:, t])
        u[1:-1, t+1] = u[1:-1, t] + dt*func(u_coch).coeffs[1:-1]
        if np.isnan(u[:, t+1]).any():
            total_err = np.nan

    if math.isnan(total_err):
        total_err = 1e5

    else:
        # evaluate errors
        u_data = u_data_T.T
        errors = u[:, time_data] - u_data

        total_err = np.mean(np.linalg.norm(errors, axis=0)**2)

    best_sol = u

    return total_err, best_sol


@ray.remote(num_cpus=2)
def eval_best_sols(individual: Callable, indlen: int, time_data: npt.NDArray,
                   u_data_T: npt.NDArray, bvalues: dict, S: SimplicialComplex,
                   num_t_points: float, num_x_points: float, dt: float,
                   u_0: C.CochainD0, penalty: dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(individual, indlen, time_data,
                                u_data_T, bvalues, S, num_t_points, num_x_points, dt, u_0)

    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(individual: Callable, indlen: int, time_data: npt.NDArray,
             u_data_T: npt.NDArray, bvalues: dict, S: SimplicialComplex,
             num_t_points: float, num_x_points: float, dt: float,
             u_0: C.CochainD0, penalty: dict) -> float:

    MSE, _ = eval_MSE_sol(individual, indlen, time_data,
                          u_data_T, bvalues, S, num_t_points, num_x_points, dt, u_0)

    return MSE


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, indlen: int, time_data: npt.NDArray,
                 u_data_T: npt.NDArray, bvalues: dict, S: SimplicialComplex,
                 num_t_points: float, num_x_points: float, dt: float,
                 u_0: C.CochainD0, penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, indlen, time_data,
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

    # problem params
    x_max = 5
    t_max = 2
    dx = 0.025
    dt = 0.001
    num_x_points = int(x_max/dx)
    num_t_points = int(t_max/dt)

    # generate mesh
    mesh, _ = util.generate_line_mesh(num_x_points, x_max)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_PDP_weights()

    # load data
    time_train, time_val, time_test, u_train_T, u_val_T, u_test_T = load_dataset(
        data_path, "npy")

    # initial guess for the solution of the problem
    x = np.linspace(0, x_max, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # initial condition
    u_0_coeffs = 2 * np.exp(-2 * (x_circ - 0.5 * x_max)**2)
    u_0 = C.CochainD0(S, u_0_coeffs)

    # boundary conditions
    nodes_BC = {'left': np.zeros(num_t_points), 'right': np.zeros(num_t_points)}

    residual_formulation = config_file["gp"]["residual_formulation"]

    # define primitive set and add primitives and terminals
    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0], C.CochainD0)
        ADF = gp.PrimitiveSetTyped("ADF", [C.CochainD0], C.CochainD1)
        pset.addADF(ADF)
    else:
        raise Exception("Only residual formulation available for this problem.")

    # add constants for MAIN
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-0.5, float, name="-1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")
    pset.addTerminal(-2., float, name="-2.")
    pset.addTerminal(10., float, name="10.")
    pset.addTerminal(0.1, float, name="0.1")

    # add constants for ADF
    ADF.addTerminal(0.5, float, name="1/2")
    ADF.addTerminal(-0.5, float, name="-1/2")
    ADF.addTerminal(-1., float, name="-1.")
    ADF.addTerminal(2., float, name="2.")
    ADF.addTerminal(-2., float, name="-2.")
    ADF.addTerminal(10., float, name="10.")
    ADF.addTerminal(0.1, float, name="0.1")

    # rename arguments
    pset.renameArguments(ARG0="u")
    ADF.renameArguments(ARG0="u")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file,  ADF=ADF)

    penalty = config_file["gp"]["penalty"]

    GPprb.store_eval_common_params({'S': S,
                                    'penalty': penalty,
                                    'bvalues': nodes_BC,
                                    'dt': dt,
                                    'num_t_points': num_t_points,
                                    'num_x_points': num_x_points,
                                    'u_0': u_0})

    params_names = ('time_data', 'u_data_T')
    datasets = {'train': [time_train, u_train_T],
                'val': [time_val, u_val_T],
                'test': [time_test, u_test_T]}
    GPprb.store_eval_dataset_params(params_names, datasets)

    GPprb.register_eval_funcs(fitness=eval_fitness.remote, error_metric=eval_MSE.remote,
                              test_sols=eval_best_sols.remote)

    # if GPprb.plot_best:
    #    GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val,
    #                           bvalues=bvalues_val, S=S, gamma=gamma, u_0=u_0,
    #                           toolbox=GPprb.toolbox)

    GPprb.register_map([gps.len_ADF])

    start = time.perf_counter()
    # from deap import creator
    # opt_string = "St1P1(cobP0(AddCP0(St1D1(flat_parD0(MFD0(SquareD0(u), -1/2))), MFP0(St1D1(cobD0(u)),0.1))))"
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPprb.run(print_log=True, seed=None,
              save_best_individual=True, save_train_fit_history=True,
              save_best_test_sols=True, X_test_param_name="time_data",
              output_path=output_path)

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
