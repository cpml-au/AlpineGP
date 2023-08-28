from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from matplotlib import tri
from deap import gp, base, creator
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


def get_boundary_values(X, left_bnd_nodes_idx, left_bnd_nodes_pos, right_bnd_nodes_idx,
                        right_bnd_pos_components, bottom_left_corner,
                        bottom_left_corner_pos):
    bvalues_X = []
    for true_curr_node_coords in X:
        right_bnd_nodes_pos = true_curr_node_coords[right_bnd_nodes_idx,
                                                    :][:, right_bnd_pos_components]

        # NOTE: without flatten it does not work properly when concatenating multiple bcs;
        # fix this so that flatten is not needed (not intuitive)
        boundary_values = {"0": (left_bnd_nodes_idx + right_bnd_nodes_idx,
                                 np.vstack((left_bnd_nodes_pos,
                                           right_bnd_nodes_pos)).flatten()),
                           ":": (bottom_left_corner, bottom_left_corner_pos)}
        bvalues_X.append(boundary_values)
    return bvalues_X


'''
def is_valid_energy(u: npt.NDArray, prb: oc.OptimizationProblem,
                    bnodes: npt.NDArray) -> bool:
    # perturb solution and check whether the gradient still vanishes
    # (i.e. constant energy)
    u_noise = u + noise*np.mean(u)
    u_noise[bnodes] = u[bnodes]
    grad_u_noise = jnp.linalg.norm(prb.gradient(u_noise))
    is_valid = grad_u_noise >= 1e-6
    return is_valid
'''


def eval_MSE_sol(func: Callable, indlen: int, X: npt.NDArray,
                 bvalues: dict, S: SimplicialComplex, gamma: float,
                 u_0: C.CochainP0) -> Tuple[float, npt.NDArray]:

    num_data, num_nodes, dim_embedded_space = X.shape

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
        penalty *= 0.5*gamma
        nodes = C.CochainP0(S, x_reshaped)
        F = C.deformation_gradient(nodes)
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
        x_flatten = prb.solve(x0=u_0.coeffs.flatten(), ftol_abs=1e-12,
                              ftol_rel=1e-12, maxeval=1000)
        # reshape x to have a tensor
        x = x_flatten.reshape(S.node_coords.shape)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):
            # check whether the energy is "admissible" (i.e. exclude constant energies)
            # valid_energy = is_valid_energy(u=x, prb=prb, bnodes=bnodes)
            # FIXME: get rid of valid_energy
            valid_energy = True

            if valid_energy:
                current_err = np.linalg.norm(x-X[i, :])**2
            else:
                current_err = math.nan
        else:
            current_err = math.nan

        if math.isnan(current_err):
            total_err = 1e5
            break

        total_err += current_err

        best_sols.append(x)

    total_err *= 1/X.shape[0]

    return 10.*total_err, best_sols


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
def plot_sol(ind: gp.PrimitiveTree, X: npt.NDArray, y: npt.NDArray,
             bvalues: dict, S: SimplicialComplex, bnodes: npt.NDArray,
             gamma: float, u_0: C.CochainP0, toolbox: base.Toolbox,
             triang: tri.Triangulation):

    indfun = toolbox.compile(expr=ind)

    _, u = eval_MSE_sol(indfun, indlen=0, X=X, y=y, bvalues=bvalues, S=S,
                        bnodes=bnodes, gamma=gamma, u_0=u_0)

    plt.figure(10, figsize=(8, 4))
    plt.clf()
    fig = plt.gcf()
    _, axes = plt.subplots(2, 3, num=10)
    for i in range(0, 3):
        axes[0, i].tricontourf(triang, u[i], cmap='RdBu', levels=20)
        pltobj = axes[1, i].tricontourf(triang, X[i], cmap='RdBu', levels=20)
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_box_aspect(1)
    plt.colorbar(pltobj, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


def stgp_linear_elasticity(config_file, output_path=None):
    global residual_formulation
    # generate mesh
    lc = 0.5
    L = 1.
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
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
    num_nodes = S.num_nodes
    num_faces = S.S[2].shape[0]
    ref_node_coords = S.node_coords

    bnd_edges_idx = S.bnd_faces_indices
    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")
    left_bnd_edges_idx = util.get_edges_for_physical_group(S, mesh, "left")
    right_bnd_edges_idx = util.get_edges_for_physical_group(S, mesh, "right")

    bottom_left_corner = left_bnd_nodes_idx.pop(0)

    left_bnd_pos_components = [0]
    right_bnd_pos_components = [0]

    left_bnd_nodes_pos = ref_node_coords[left_bnd_nodes_idx,
                                         :][:, left_bnd_pos_components]
    bottom_left_corner_pos = ref_node_coords[bottom_left_corner, :]

    # FIXME: just to initialize ref_metric_contravariant. Write a routine in simplex that does it
    _ = S.get_deformation_gradient(ref_node_coords)

    # extract boundary values
    bvalues_train = get_boundary_values(X_train,
                                        left_bnd_nodes_idx,
                                        left_bnd_nodes_pos,
                                        right_bnd_nodes_idx,
                                        right_bnd_pos_components,
                                        bottom_left_corner,
                                        bottom_left_corner_pos)
    bvalues_val = get_boundary_values(X_val,
                                      left_bnd_nodes_idx,
                                      left_bnd_nodes_pos,
                                      right_bnd_nodes_idx,
                                      right_bnd_pos_components,
                                      bottom_left_corner,
                                      bottom_left_corner_pos)
    bvalues_test = get_boundary_values(X_test,
                                       left_bnd_nodes_idx,
                                       left_bnd_nodes_pos,
                                       right_bnd_nodes_idx,
                                       right_bnd_pos_components,
                                       bottom_left_corner,
                                       bottom_left_corner_pos)

    # penalty parameter for the Dirichlet bcs
    gamma = 10000.

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
        pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP2T], float)

    # define ADF
    ADF = gp.PrimitiveSetTyped("epsilon", [C.CochainP2T], C.CochainP2T)
    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1.")
    pset.addTerminal(2., float, name="2.")
    pset.addTerminal(10., float, name="10.")
    pset.addTerminal(0.1, float, name="0.1")

    identity = jnp.stack([jnp.identity(2)]*num_faces)
    identity_coch = C.CochainP2T(S, identity)
    pset.addTerminal(identity_coch, C.CochainP2T, name="I")
    pset.addADF(ADF)

    # rename arguments
    pset.renameArguments(ARG0="F")

    # create symbolic regression problem instance
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file, ADF=ADF)

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

    '''
    if GPprb.plot_best:
        triang = tri.Triangulation(S.node_coords[:, 0], S.node_coords[:, 1], S.S[2])
        GPprb.toolbox.register("plot_best_func", plot_sol, X=X_val, y=y_val,
                               bvalues=bvalues_val, S=S, bnodes=bnodes,
                               gamma=gamma, u_0=u_0,
                               toolbox=GPprb.toolbox, triang=triang)

    '''
    GPprb.register_map([len])

    start = time.perf_counter()
    # epsilon = "SubP2T(MulFP2T(AddP2T(F, tranP2T(F)), 1/2), I)"
    # opt_string_eps = "Add(MulF(2., InnP2T(epsilon, epsilon)), MulF(10., InnP2T(MulVP2VT(trP2T(epsilon), I), epsilon)))"
    # opt_string_eps = "InnP2T(epsilon, epsilon)"
    # opt_string = opt_string_eps.replace("epsilon", epsilon)
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
