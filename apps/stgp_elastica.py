import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, Tuple
import jax.numpy as jnp
from jax import grad, Array
from deap import base, gp, tools
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_line_mesh, build_complex_from_mesh
from dctkit.dec import cochain as C
from dctkit import config
from dctkit.math.opt import optctrl as oc
import dctkit as dt
from alpine.data.util import load_dataset
from alpine.data.elastica.elastica_dataset import data_path
from alpine.gp import gpsymbreg as gps
from apps.util import get_positions_from_angles
import matplotlib.pyplot as plt
import math
import sys
import yaml
import time
import ray

residual_formulation = False

# choose precision and whether to use GPU or CPU
#  needed for context of the plots at the end of the evolution
config()

NUM_NODES = 11
LENGTH = 1.


def get_angles_initial_guesses(x: list, y: list) -> Dict:
    theta_in_all_list = []
    for i in range(3):
        x_current = x[i]
        y_current = y[i]
        theta_in_init = np.ones((x_current.shape[0], x_current.shape[1]-2),
                                dtype=dt.float_dtype)
        const_angles = np.arctan(
            (y_current[:, -1] - y_current[:, 1])/(x_current[:, -1] - x_current[:, 1]))
        theta_0_current = np.diag(const_angles) @ theta_in_init
        theta_in_all_list.append(theta_0_current)

    theta_in_all_dict = dict([('train', theta_in_all_list[0]),
                              ('val', theta_in_all_list[1]),
                              ('test', theta_in_all_list[2])])
    return theta_in_all_dict


def is_valid_energy(theta_in: npt.NDArray, theta: npt.NDArray,
                    prb: oc.OptimizationProblem) -> bool:
    dim = len(theta_in)
    noise = 0.0001*np.ones(dim).astype(dt.float_dtype)
    theta_in_noise = theta_in + noise
    theta_noise = prb.solve(x0=theta_in_noise, maxeval=500, ftol_abs=1e-12,
                            ftol_rel=1e-12)
    is_valid = np.allclose(theta, theta_noise, rtol=1e-6, atol=1e-6)
    return is_valid


class Objectives():
    def __init__(self, S: SimplicialComplex) -> None:
        self.S = S

    def set_residual(self, func: Callable) -> None:
        """Set the energy function to be used for the computation of the objective
        function."""
        self.residual = func

    def set_energy_func(self, func: Callable) -> None:
        """Set the energy function to be used for the computation of the objective
        function."""
        self.energy_func = func

    # elastic energy including Dirichlet BC by elimination of the prescribed dofs
    def total_energy(self, theta_vec: npt.NDArray, FL2_EI0: float,
                     theta_0: npt.NDArray) -> Array:
        # extend theta on the boundary w.r.t boundary conditions
        theta_vec = jnp.insert(theta_vec, 0, theta_0)
        theta = C.CochainD0(self.S, theta_vec)
        FL2_EI0_coch = C.CochainD0(
            self.S, FL2_EI0*jnp.ones(self.S.num_nodes-1, dtype=dt.float_dtype))
        if residual_formulation:
            residual = self.residual(theta, FL2_EI0_coch)
            energy = jnp.linalg.norm(residual.coeffs[1:])**2
        else:
            energy = self.energy_func(theta, FL2_EI0_coch)
        return energy

    # state function: stationarity conditions for the total energy
    def total_energy_grad(self, x: npt.NDArray, theta_0: float) -> Array:
        theta = x[:-1]
        FL2_EI0 = x[-1]
        if residual_formulation:
            # FIXME: not sure why we are not applying grad to total_energy
            theta_vec = jnp.insert(theta, 0, theta_0)
            theta = C.CochainD0(self.S, theta_vec)
            FL2_EI0_coch = C.CochainD0(
                self.S, FL2_EI0*jnp.ones(self.S.num_nodes-1, dtype=dt.float_dtype))
            return self.residual(theta, FL2_EI0_coch).coeffs[1:]
        else:
            return grad(self.total_energy)(theta, FL2_EI0, theta_0)

    # objective function for the parameter EI0 identification problem
    def MSE_theta(self, x: npt.NDArray, theta_true: npt.NDArray) -> Array:
        theta = x[:-1]
        theta = jnp.insert(theta, 0, theta_true[0])
        return jnp.sum(jnp.square(theta-theta_true))


@ray.remote(num_cpus=2)
def tune_EI0(func: Callable, EI0: float, indlen: int, FL2: float,
             EI0_guess: float, theta_guess: npt.NDArray,
             theta_true: npt.NDArray, S: SimplicialComplex) -> float:

    # number of unknowns angles
    dim = S.num_nodes-2
    obj = Objectives(S=S)
    if residual_formulation:
        obj.set_residual(func)
    else:
        obj.set_energy_func(func)

    # prescribed angle at x=0
    theta_0 = theta_true[0]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    constraint_args = {'theta_0': theta_0}
    obj_args = {'theta_true': theta_true}

    prb = oc.OptimalControlProblem(objfun=obj.MSE_theta, statefun=obj.total_energy_grad,
                                   state_dim=dim, nparams=S.num_nodes-1,
                                   constraint_args=constraint_args, obj_args=obj_args)

    def get_bounds():
        lb = -100*np.ones(dim+1, dt.float_dtype)
        ub = 100*np.ones(dim+1, dt.float_dtype)
        lb[-1] = -100
        ub[-1] = -1e-3
        return (lb, ub)

    prb.get_bounds = get_bounds

    FL2_EI0 = FL2/EI0_guess

    x0 = np.append(theta_guess, FL2_EI0)

    x = prb.solve(x0=x0, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

    # theta = x[:-1]
    FL2_EI0 = x[-1]

    EI0 = FL2/FL2_EI0

    # if optimization failed, set negative EI0
    if not (prb.last_opt_result == 1 or prb.last_opt_result == 3 or
            prb.last_opt_result == 4):
        EI0 = -1.

    return EI0


def eval_MSE_sol(func: Callable, EI0: float, indlen: int,
                 thetas_true: npt.NDArray, Fs: npt.NDArray, S: SimplicialComplex,
                 theta_in_all: npt.NDArray) -> Tuple[float, npt.NDArray]:

    # number of unknown angles
    dim = S.num_nodes-2

    total_err = 0.

    obj = Objectives(S=S)
    if residual_formulation:
        obj.set_residual(func)
    else:
        obj.set_energy_func(func)

    X_dim = thetas_true.shape[0]
    best_theta = np.zeros((X_dim, S.num_nodes-1), dtype=dt.float_dtype)

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    if EI0 < 0.:
        total_err = 40.
    else:
        for i, theta_true in enumerate(thetas_true):
            # extract prescribed value of theta at x = 0 from the dataset
            theta_0 = theta_true[0]

            theta_in = theta_in_all[i, :]

            FL2 = Fs[i]
            FL2_EI0 = FL2/EI0

            prb = oc.OptimizationProblem(
                dim=dim, state_dim=dim, objfun=obj.total_energy)
            args = {'FL2_EI0': FL2_EI0, 'theta_0': theta_0}
            prb.set_obj_args(args)
            theta = prb.solve(x0=theta_in, maxeval=500, ftol_abs=1e-12, ftol_rel=1e-12)

            # check whether the energy is "admissible" (i.e. exclude constant energies
            # and energies with minima that are too sensitive to the initial guess)
            valid_energy = is_valid_energy(theta_in=theta_in, theta=theta, prb=prb)

            if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                    or prb.last_opt_result == 4) and valid_energy:
                x = np.append(theta, FL2_EI0)
                fval = float(obj.MSE_theta(x, theta_true))
            else:
                fval = math.nan

            if math.isnan(fval):
                total_err = 40.
                break

            total_err += fval

            # extend theta
            theta = np.insert(theta, 0, theta_0)

            # update best_theta
            best_theta[i, :] = theta

    total_err *= 1/(X_dim)

    # round total_err to 5 decimal digits
    total_err = float("{:.5f}".format(total_err))

    return 10.*total_err, best_theta

# the following remote functions must have the same signature


@ray.remote(num_cpus=2)
def eval_best_sols(energy_func: Callable, EI0: float, indlen: int,
                   thetas_true: npt.NDArray, Fs: npt.NDArray, S: SimplicialComplex,
                   theta_in_all: npt.NDArray, penalty: Dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(energy_func, EI0, indlen, thetas_true, Fs, S,
                                theta_in_all)
    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(energy_func: Callable, EI0: float, indlen: int, thetas_true: npt.NDArray,
             Fs: npt.NDArray, S: SimplicialComplex, theta_in_all: npt.NDArray,
             penalty: Dict) -> float:

    MSE, _ = eval_MSE_sol(energy_func, EI0, indlen, thetas_true, Fs, S, theta_in_all)

    return MSE


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, EI0: float, indlen: int,
                 thetas_true: npt.NDArray, Fs: npt.NDArray, S: SimplicialComplex,
                 theta_in_all: npt.NDArray, penalty: Dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, EI0, indlen,
                                thetas_true, Fs, S, theta_in_all)

    # penalty terms on length
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


def plot_sol(ind: gp.PrimitiveTree, thetas_true: npt.NDArray, Fs: npt.NDArray,
             toolbox: base.Toolbox, S: SimplicialComplex, theta_in_all: npt.NDArray):

    indfun = toolbox.compile(expr=ind)

    _, best_sol_all = eval_MSE_sol(indfun, ind.EI0, indlen=0,
                                   thetas_true=thetas_true, Fs=Fs, S=S,
                                   theta_in_all=theta_in_all)

    plt.figure(1, figsize=(10, 4))
    plt.clf()
    dim = thetas_true.shape[0]
    fig = plt.gcf()
    _, axes = plt.subplots(1, dim, num=1)
    # get the x,y coordinates LIST of the best and of the true
    x_curr, y_curr = get_positions_from_angles((best_sol_all,))
    x_true, y_true = get_positions_from_angles((thetas_true,))
    for i in range(dim):
        # plot the results
        axes[i].plot(x_true[0][i, :], y_true[0][i, :], 'r')
        axes[i].plot(x_curr[0][i, :], y_curr[0][i, :], 'b')

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.1)


def stgp_elastica(config_file_data, output_path=None):
    global residual_formulation
    thetas_train, thetas_val, thetas_test, Fs_train, Fs_val, Fs_test = load_dataset(
        data_path, "csv")

    # TODO: how can we extract these numbers from the dataset (especially length)?
    mesh, _ = generate_line_mesh(num_nodes=NUM_NODES, L=LENGTH)
    S = build_complex_from_mesh(mesh)
    S.get_hodge_star()

    x_all, y_all = get_positions_from_angles(
        (thetas_train, thetas_val, thetas_test))

    theta_in_all = get_angles_initial_guesses(x_all, y_all)

    residual_formulation = config_file["gp"]["residual_formulation"]

    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, C.CochainD0], C.CochainD0)
    else:
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, C.CochainD0], float)

    # add internal cochain as a terminal
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0.
    internal_vec[-1] = 0.
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")
    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")

    pset.renameArguments(ARG0="theta")
    pset.renameArguments(ARG1="FL2_EI0")

    GPprb = gps.GPSymbolicRegressor(pset=pset, config_file_data=config_file_data)

    penalty = config_file_data["gp"]['penalty']

    # store shared objects refs: does it work without ray?
    GPprb.store_eval_common_params({'S': S, 'penalty': penalty})
    param_names = ('thetas_true', 'Fs', 'theta_in_all')
    datasets = {'train': [thetas_train, Fs_train, theta_in_all['train']],
                'val': [thetas_val, Fs_val, theta_in_all['val']],
                'test': [thetas_test, Fs_test, theta_in_all['test']]}
    GPprb.store_datasets_params(param_names, datasets)

    GPprb.register_eval_funcs(fitness=eval_fitness.remote, error_metric=eval_MSE.remote,
                              eval_sol=eval_best_sols.remote)

    # register custom functions
    GPprb.toolbox.register("evaluate_EI0", tune_EI0.remote, FL2=Fs_train[0],
                           EI0_guess=1.,
                           theta_guess=theta_in_all['train'][0, :],
                           theta_true=thetas_train[0, :],
                           S=GPprb.data_store['common']['S'])

    if GPprb.plot_best:
        GPprb.toolbox.register("plot_best_func", plot_sol, thetas_true=thetas_val,
                               Fs=Fs_val, toolbox=GPprb.toolbox, S=S,
                               theta_in_all=theta_in_all['val'])

    # opt_string = "SubF(MulF(1/2, InnP0(CMulP0(int_coch, St1D1(cobD0(theta))),
    # CMulP0(int_coch, St1D1(cobD0(theta))))), InnD0(FL2_EI0, SinD0(theta)))"
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    feature_extractors = [lambda ind: ind.EI0, len]

    GPprb.__register_map(feature_extractors)

    def evaluate_EI0s(pop):
        if not hasattr(pop[0], "EI0"):
            for ind in pop:
                ind.EI0 = 1.

        EI0s = GPprb.toolbox.map(GPprb.toolbox.evaluate_EI0, pop)

        for ind, EI0 in zip(pop, EI0s):
            ind.EI0 = EI0

    def print_EI0(pop):
        best = tools.selBest(pop, k=1)[0]
        print("The best individual's EI0 is: ", best.EI0)

    start = time.perf_counter()

    GPprb.__run(print_log=True, seed=None, save_train_fit_history=True,
                save_best_individual=True, save_best_test_sols=True,
                X_test_param_name='thetas_true', output_path=output_path,
                preprocess_fun=evaluate_EI0s, callback_fun=print_EI0)

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

    stgp_elastica(config_file, output_path)
