import numpy as np
import jax.numpy as jnp
from deap import base, gp
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl
from dctkit.dec import cochain as C
import dctkit as dt
import math
import sys
import yaml

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

    def total_energy(self, theta_vec: np.array, EI0: np.array, theta_in: np.array) -> float:
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
        energy = self.energy_func(theta, EI0)
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

    for i, theta_true in enumerate(X):
        # extract boundary value in 0
        theta_in = theta_true[0]

        # set extra args for bilevel program
        constraint_args = (theta_in)
        obj_args = (theta_true)

        # set bilevel problem
        prb = optctrl.OptimalControlProblem(objfun=obj_fun_theta,
                                            state_en=obj.total_energy,
                                            state_dim=S.num_nodes-2,
                                            constraint_args=constraint_args,
                                            obj_args=obj_args)

        theta, EI0, fval = prb.run(theta_0, EI0_0, tol=1e-5)
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
        current_err_EI0 = np.linalg.norm(EI0 - y[i])
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
        obj_val = total_err
    elif penalty["method"] == "length":
        # penalty terms on length
        objval = total_err + penalty["reg_param"]*len(individual)
    else:
        # no penalty
        objval = total_err
    return objval,


def stgp_elastica(config_file):
    pass


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))
    stgp_elastica(config_file)
