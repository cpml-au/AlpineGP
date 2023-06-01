<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/alpine.svg?branch=main)](https://cirrus-ci.com/github/<USER>/alpine)
[![ReadTheDocs](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/alpine/main.svg)](https://coveralls.io/r/<USER>/alpine)
[![PyPI-Server](https://img.shields.io/pypi/v/alpine.svg)](https://pypi.org/project/alpine/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/alpine.svg)](https://anaconda.org/conda-forge/alpine)
[![Monthly Downloads](https://pepy.tech/badge/alpine/month)](https://pepy.tech/project/alpine)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/alpine)
-->


# alpine

_Alpine_ is a Python library that helps to build algorithms that learn *symbolic* models
of _physical systems_ starting from data. It combines the _Discrete Calculus_ framework
implemented in the library [`dctkit`](https://github.com/alucantonio/dctkit) with the
Strongly-Typed Genetic Programming approach to symbolic regression _via_ the
[`DEAP`](https://github.com/alucantonio/DEAP) library.

## Installation

Dependencies should be installed within a `conda` environment. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
$ conda env create -f environment.yaml
```

Otherwise, update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

```bash
$ pip install -e .
```

to install a development version of the `alpine` library.

Running the tests:

```bash
$ tox
```

Generating the docs:

```bash
$ tox -e docs
```

## Usage
Prototype problem: identifying the Poisson equation in 1D.

1. Import the relevant libraries
```python
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from deap import gp
from alpine.gp import gpsymbreg as gps
from alpine.gp import primitives
from dctkit import config
import dctkit
import networkx as nx
from ray.util.multiprocessing import Pool
import ray
import numpy as np
import math
import yaml
from typing import Tuple, Callable
import numpy.typing as npt
```

2. Define the function to compute the fitness of an individual (model expression tree) 
```python
def eval_MSE(residual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
             S: SimplicialComplex, u_0: C.CochainP0, return_best_sol: bool = False) -> float:

    num_nodes = X.shape[1]

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    # objective: squared norm of the residual of the equation + penalty on Dirichlet 
    # boundary condition on the first node
    def obj(x, y):
        penalty = 100.*x[0]**2
        u = C.CochainP0(S, x)
        f = C.CochainP0(S, y)
        r = residual(u, f)
        total_energy = C.inner_product(r, r) + penalty
        return total_energy

    prb = oc.OptimizationProblem(dim=num_nodes, state_dim=num_nodes, objfun=obj)

    total_err = 0.

    best_sols = []

    for i, vec_y in enumerate(y):
        # set additional arguments of the objective function (apart from the vector of unknowns)
        args = {'y': vec_y}
        prb.set_obj_args(args)

        # minimize the objective
        x = prb.run(x0=u_0.coeffs, ftol_abs=1e-12, ftol_rel=1e-12, maxeval=1000)

        if (prb.last_opt_result == 1 or prb.last_opt_result == 3
                or prb.last_opt_result == 4):

            current_err = np.linalg.norm(x-X[i, :])**2
        else:
            current_err = math.nan

        if math.isnan(current_err):
            total_err = 1e5
            break

        total_err += current_err

        best_sols.append(x)

    if return_best_sol:
        return best_sols

    total_err *= 1/X.shape[0]

    return total_err

@ray.remote
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 S: SimplicialComplex, u_0: C.CochainP0, penalty: dict) -> Tuple[float, ]:

    objval = 0.

    total_err = eval_MSE(individual, indlen, X, y, S, u_0)

    # add penalty on length of the tree to promote simpler solutions
    objval = total_err + penalty["reg_param"]*indlen

    return objval,
```

3. Define the main function to run the symbolic regression
```python
def stgp_poisson():
    with open("ex1.yaml") as config_file:
        config_file_data = yaml.safe_load(config_file)

    # generate mesh and dataset
    S_1, x = generate_1_D_mesh(num_nodes=11, L=1.)
    S = SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    num_nodes = S.num_nodes

    # generate training and test datasets
    # exact solution = x² 
    u = C.CochainP0(S, np.array(x[:,0]**2, dtype=dctkit.float_dtype))
    # compute source term such that u solves the discrete Poisson equation 
    # Delta u + f = 0, where Delta is the discrete Laplace-de Rham operator
    f = C.laplacian(u)
    f.coeffs *= -1.
    X_train = np.array([u.coeffs],dtype=dctkit.float_dtype)
    y_train = np.array([f.coeffs], dtype=dctkit.float_dtype)

    # initial guess for the unknown of the Poisson problem (cochain of nodals values)
    u_0_vec = np.zeros(num_nodes, dtype=dctkit.float_dtype)
    u_0 = C.CochainP0(S, u_0_vec)

    # define primitive set for the residual of the discrete Poisson equation
    pset = gp.PrimitiveSetTyped("RESIDUAL", [C.CochainP0, C.CochainP0], C.CochainP0)

    # rename arguments of the residual
    pset.renameArguments(ARG0="u")
    pset.renameArguments(ARG1="f")

    # set parameters of the symbolic regression problem from config file
    # also create toolbox needed for Genetic Programming algorithm
    GPproblem_settings, GPproblem_run, GPproblem_extra = gps.load_config_data(
        config_file_data=config_file_data, pset=pset)
    toolbox = GPproblem_settings['toolbox']
    penalty = GPproblem_extra['penalty']

    # add primitives to the primitive set, as specified in the config file
    primitives.addPrimitivesToPset(pset, GPproblem_settings['primitives'])

    GPproblem_settings.pop('primitives')

    ray.init()

    # store shared objects refs
    S_ref = ray.put(S)
    u_0_ref = ray.put(u_0)
    penalty_ref = ray.put(penalty)
    X_train_ref = ray.put(X_train)
    y_train_ref = ray.put(y_train)

    # set arguments for evaluate functions
    args_train = {'X': X_train_ref, 'y': y_train_ref, 'S': S_ref, 'u_0': u_0_ref, 
                  'penalty': penalty_ref}

    # register functions for fitness/MSE evaluation on different datasets
    toolbox.register("evaluate_train", eval_fitness.remote, **args_train)

    # create symbolic regression problem instance
    GPproblem = gps.GPSymbRegProblem(pset=pset, **GPproblem_settings)

    # MULTIPROCESSING MAP FUNCTION----------------------------------------------------
    pool = Pool()

    def ray_mapper(f, individuals, toolbox):
        # We are not duplicating global scope on workers so we need to use the toolbox
        # Transform the tree expression in a callable function
        runnables = [toolbox.compile(expr=ind) for ind in individuals]
        lenghts = [len(ind) for ind in individuals]
        fitnesses = ray.get([f(*args) for args in zip(runnables, lenghts)])
        return fitnesses

    GPproblem.toolbox.register("map", ray_mapper, toolbox=GPproblem.toolbox)
    # --------------------------------------------------------------------------------

    GPproblem.run(plot_history=False, print_log=True, seed=None, **GPproblem_run)

    best = GPproblem.best

    # compute the solution corresponding to the best individual
    u_best = eval_MSE(GPproblem.toolbox.compile(expr=best), len(str(best)),
                          X_train, y_train, S=S, u_0=u_0, return_best_sol=True)[0]

    pool.close()
    ray.shutdown()

    plt.figure()
    plt.plot(x[:,0], u.coeffs)
    plt.plot(x[:,0], u_best, "ro")
    plt.show()

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
```

The complete example (notebook) can be found in the `examples` directory.