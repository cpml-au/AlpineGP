from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from matplotlib.pyplot import triplot
# from matplotlib import tri
from deap import gp, base
from deap import creator
from alpine.data.util import load_dataset
from alpine.data.linear_elasticity.linear_elasticity_dataset import data_path
from dctkit.mesh import util
from alpine.gp import gpsymbreg as gps
from dctkit import config
import dctkit

import ray

import numpy as np
import jax.numpy as jnp
import math
import time
import sys
import yaml
from typing import Tuple, Callable
import numpy.typing as npt
import pygmsh

residual_formulation = False

# choose precision and whether to use GPU or CPU
# needed for context of the plots at the end of the evolution
config()


def get_boundary_values(X, y, ref_node_coords, boundary_nodes_info):
    bvalues_X = []
    right_bnd_nodes_idx = boundary_nodes_info['right_bnd_nodes_idx']
    left_bnd_nodes_idx = boundary_nodes_info['left_bnd_nodes_idx']
    up_bnd_nodes_idx = boundary_nodes_info['up_bnd_nodes_idx']
    down_bnd_nodes_idx = boundary_nodes_info['down_bnd_nodes_idx']
    for i, data_label in enumerate(y):
        true_curr_node_coords = X[i, :, :]
        if data_label == "pure_tension":
            bot_left_corn_idx = left_bnd_nodes_idx.index(0)
            bottom_left_corner = left_bnd_nodes_idx[bot_left_corn_idx]
            left_bnd_nodes_without_corner = left_bnd_nodes_idx[:bot_left_corn_idx] + \
                left_bnd_nodes_idx[(bot_left_corn_idx+1):]

            left_bnd_pos_components = [0]
            right_bnd_pos_components = [0]

            left_bnd_nodes_pos = ref_node_coords[left_bnd_nodes_without_corner,
                                                 :][:, left_bnd_pos_components]
            bottom_left_corner_pos = ref_node_coords[bottom_left_corner, :]
            right_bnd_nodes_pos = true_curr_node_coords[right_bnd_nodes_idx,
                                                        :][:, right_bnd_pos_components]

            # NOTE: without flatten it does not work properly when concatenating
            # multiple bcs; fix this so that flatten is not needed (not intuitive)
            boundary_values = {"0": (left_bnd_nodes_without_corner+right_bnd_nodes_idx,
                                     np.vstack((left_bnd_nodes_pos,
                                               right_bnd_nodes_pos)).flatten()),
                               ":": (bottom_left_corner, bottom_left_corner_pos)}
        elif data_label == "pure_shear":
            up_bnd_nodes_pos_x = true_curr_node_coords[up_bnd_nodes_idx, 0]
            up_bnd_nodes_pos_y = ref_node_coords[up_bnd_nodes_idx, 1]
            up_bnd_pos = np.zeros((len(up_bnd_nodes_idx), 3))
            up_bnd_pos[:, 0] = up_bnd_nodes_pos_x
            up_bnd_pos[:, 1] = up_bnd_nodes_pos_y
            down_bnd_pos = ref_node_coords[down_bnd_nodes_idx, :]
            left_bnd_pos = true_curr_node_coords[left_bnd_nodes_idx, :]
            right_bnd_pos = true_curr_node_coords[right_bnd_nodes_idx, :]
            bnodes = left_bnd_nodes_idx + right_bnd_nodes_idx + \
                up_bnd_nodes_idx + down_bnd_nodes_idx
            bvalues = np.vstack((left_bnd_pos, right_bnd_pos,
                                 up_bnd_pos, down_bnd_pos))
            boundary_values = {":": (bnodes, bvalues)}

        bvalues_X.append(boundary_values)
    return bvalues_X


def eval_MSE_sol(func: Callable, indlen: int, X: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, gamma: float,
                 u_0: C.CochainP0) -> Tuple[float, npt.NDArray]:

    num_data, num_nodes, dim_embedded_space = X.shape

    num_faces = S.S[2].shape[0]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # create objective function and set its energy function
    def total_energy(x, curr_bvalues):
        x_reshaped = x.reshape(S.node_coords.shape)
        penalty = 0.
        for key in curr_bvalues:
            idx, values = curr_bvalues[key]
            if key == ":":
                penalty += jnp.sum((x_reshaped[idx, :] - values)**2)
            else:
                penalty += jnp.sum((x_reshaped[idx, int(key)] - values)**2)
        penalty *= gamma
        nodes = C.CochainP0(S, x_reshaped)
        F = C.deformation_gradient(nodes)
        # identity = jnp.stack([jnp.identity(2)]*num_faces)
        # I = C.CochainD0T(S, identity)
        # epsilon = C.sub(C.scalar_mul(C.add(F, C.transpose(F)), 0.5), I)
        # if residual_formulation:
        #    total_energy = C.inner_product(func(c, fk), func(c, fk)) + penalty
        # else:
        total_energy = func(F) + penalty
        return total_energy

    prb = oc.OptimizationProblem(dim=num_nodes*dim_embedded_space,
                                 state_dim=num_nodes*dim_embedded_space,
                                 objfun=total_energy)

    total_err = 0.

    best_sols = []

    # for i, x in enumerate(X):
    for i in range(num_data):
        # extract current bvalues
        curr_bvalues = bvalues[i]

        # set current bvalues and vec_y for the Poisson problem
        # args = {'vec_y': vec_y, 'vec_bvalues': vec_bvalues}
        args = {'curr_bvalues': curr_bvalues}
        prb.set_obj_args(args)

        # minimize the objective
        x_flatten = prb.solve(x0=u_0.coeffs.flatten(), maxeval=1000,
                              ftol_abs=1e-12, ftol_rel=1e-12)
        # reshape x to have a tensor
        # x = x_flatten.reshape(S.node_coords.shape)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):

            current_err = np.linalg.norm(x_flatten-X[i, :].flatten())**2
            x_reshaped = x_flatten.reshape(S.node_coords.shape)
            curr_nodes = C.CochainP0(S, x_reshaped)
            F = C.deformation_gradient(curr_nodes)
            W = jnp.stack(
                [jnp.array([[0, jnp.e], [-jnp.e, 0]])]*num_faces)
            F_plus_W = C.CochainD0(S, F.coeffs + W)
            current_err += (func(F) - func(F_plus_W))**2
        else:
            current_err = math.nan

        if math.isnan(current_err):
            total_err = 1e5
            break

        total_err += current_err

        best_sols.append(x_flatten.reshape(S.node_coords.shape))

    total_err *= 1/X.shape[0]

    return 1000.*total_err, best_sols


@ray.remote(num_cpus=2)
def eval_best_sols(individual: Callable, indlen: int, X: npt.NDArray,
                   bvalues: dict, S: SimplicialComplex,
                   gamma: float, u_0: npt.NDArray, penalty: dict) -> npt.NDArray:

    _, best_sols = eval_MSE_sol(individual, indlen, X, bvalues, S,
                                gamma, u_0)

    return best_sols


@ray.remote(num_cpus=2)
def eval_MSE(individual: Callable, indlen: int, X: npt.NDArray,
             bvalues: dict, S: SimplicialComplex,
             gamma: float, u_0: npt.NDArray, penalty: dict) -> float:

    MSE, _ = eval_MSE_sol(individual, indlen, X, bvalues, S, gamma, u_0)

    return MSE


@ray.remote(num_cpus=2)
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, gamma: float,
                 u_0: npt.NDArray, penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, indlen, X, bvalues, S, gamma, u_0)

    # penalty terms on length
    objval = total_err + penalty["reg_param"]*indlen

    return objval,


# Plot best solution
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, bvalues: dict,
             S: SimplicialComplex, gamma: float, u_0: C.CochainP0,
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


def stgp_linear_elasticity(config_file, output_path=None):
    global residual_formulation
    # generate mesh
    lc = 0.1
    L = 1.
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        geom.add_physical(p.lines[0], label="down")
        geom.add_physical(p.lines[2], label="up")
        geom.add_physical(p.lines[1], label="right")
        geom.add_physical(p.lines[3], label="left")
        mesh = geom.generate_mesh()

    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPD_weights()
    S.get_flat_DPP_weights()

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_path, "npy")

    # set bc
    num_faces = S.S[2].shape[0]
    ref_node_coords = S.node_coords

    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")
    down_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "down")
    up_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "up")

    # FIXME: just to initialize ref_metric_contravariant.
    # Write a routine in simplex that does it
    _ = S.get_deformation_gradient(ref_node_coords)

    # define a dictionary containing boundary nodes information (needed to set properly
    #  boundary_values)
    boundary_nodes_info = {'left_bnd_nodes_idx': left_bnd_nodes_idx,
                           'right_bnd_nodes_idx': right_bnd_nodes_idx,
                           'up_bnd_nodes_idx': up_bnd_nodes_idx,
                           'down_bnd_nodes_idx': down_bnd_nodes_idx}

    # extract boundary values
    bvalues_train = get_boundary_values(X=X_train,
                                        y=y_train,
                                        ref_node_coords=ref_node_coords,
                                        boundary_nodes_info=boundary_nodes_info)
    bvalues_val = get_boundary_values(X=X_val,
                                      y=y_val,
                                      ref_node_coords=ref_node_coords,
                                      boundary_nodes_info=boundary_nodes_info)
    bvalues_test = get_boundary_values(X=X_test,
                                       y=y_test,
                                       ref_node_coords=ref_node_coords,
                                       boundary_nodes_info=boundary_nodes_info)

    # penalty parameter for the Dirichlet bcs
    gamma = 1000000.

    # initial guess for the solution of the problem
    u_0 = C.CochainP0(S, ref_node_coords)

    residual_formulation = config_file["gp"]["residual_formulation"]

    # define primitive set and add primitives and terminals
    if residual_formulation:
        print("Using residual formulation.")
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], C.Cochain)
        # ones cochain
        pset.addTerminal(C.Cochain(S.num_nodes, True, S, np.ones(
            S.num_nodes, dtype=dctkit.float_dtype)), C.Cochain, name="F")
    else:
        pset = gp.PrimitiveSetTyped("F", [C.CochainD0T], float)

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")
    pset.addTerminal(10., float, name="10.")
    pset.addTerminal(0.1, float, name="0.1")

    identity = jnp.stack([jnp.identity(2)]*num_faces)
    identity_coch = C.CochainD0T(S, identity)
    pset.addTerminal(identity_coch, C.CochainD0T, name="I")

    # rename arguments
    pset.renameArguments(ARG0="F")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file)

    penalty = config_file["gp"]["penalty"]

    GPprb.store_eval_common_params({'S': S, 'penalty': penalty,
                                    'gamma': gamma, 'u_0': u_0})

    params_names = ('X', 'bvalues')
    datasets = {'train': [X_train, bvalues_train],
                'val': [X_val, bvalues_val],
                'test': [X_test, bvalues_test]}
    GPprb.store_eval_dataset_params(params_names, datasets)

    GPprb.register_eval_funcs(fitness=eval_fitness.remote, error_metric=eval_MSE.remote,
                              test_sols=eval_best_sols.remote)

    if GPprb.plot_best:
        GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val,
                               bvalues=bvalues_val, S=S, gamma=gamma, u_0=u_0,
                               toolbox=GPprb.toolbox)

    GPprb.register_map([len])

    start = time.perf_counter()
    # epsilon = "SubCD0T(symD0T(F), I)"
    # opt_string_eps = "AddF(MulF(2., InnD0T(epsilon, epsilon)), MulF(10., InnD0T(MvD0VT(trD0T(epsilon), I), epsilon)))"
    # opt_string = opt_string_eps.replace("epsilon", epsilon)
    # opt_string = "InnD0T(AddCD0T(MvD0VT(trD0T(SubCD0T(F, I)), AddCD0T(AddCD0T(I, I),
    # AddCD0T(AddCD0T(I, I), I))), SubCD0T(F, I)), SubCD0T(F, I))"
    # opt_individ = creator.Individual.from_string(opt_string, pset)
    # seed = [opt_individ]

    GPprb.run(print_log=True, seed=None,
              save_best_individual=True, save_train_fit_history=True,
              save_best_test_sols=True, X_test_param_name="X",
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

    stgp_linear_elasticity(config_file, output_path)
