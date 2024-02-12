[![Documentation Status](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/latest/?badge=latest)

# AlpineGP

_AlpineGP_ is a Python library that helps to build algorithms that learn *symbolic* models
of _physical systems_ starting from data. It combines the _Discrete Calculus_ framework
implemented in the library [`dctkit`](https://github.com/alucantonio/dctkit) with the
Strongly-Typed Genetic Programming approach to _symbolic regression_ via the
[`DEAP`](https://github.com/alucantonio/DEAP) library.

## Installation

Dependencies should be installed within a `conda` environment. We recommend using
[`mamba`](https://github.com/mamba-org/mamba) since it is much faster than `conda` at
solving the environment and downloading the dependencies. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
$ mamba env create -f environment.yaml
```

Otherwise, update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

```bash
$ pip install -e .
```

to install a development version of the library.

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

1. Import the relevant libraries.
```python
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_line_mesh, build_complex_from_mesh
from dctkit.math.opt import optctrl as oc
import matplotlib.pyplot as plt
from deap import gp
from alpine.gp import gpsymbreg as gps
from dctkit import config
import dctkit
import numpy as np
import ray
import math
import yaml
from typing import Tuple, Callable, List
import numpy.typing as npt
```

2. Define the function to compute the fitness of an individual (model expression tree). 
```python
def eval_MSE_sol(residual: Callable, X: npt.NDArray, y: npt.NDArray,
                 S: SimplicialComplex, u_0: C.CochainP0) -> float:

    # need to call config again before using JAX in energy evaluations to make sure that
    # the current worker has initialized JAX
    config()

    total_err = 0.
    best_sols = []

    # define an optimization problem with objective function the squared norm of the 
    # residual
    # ...
    # ...

    # loop over the dataset (training or test)
    for i, vec_y in enumerate(y):
        # compute the minimizer of the optimization problem for given inputs (X)
        # ...
        # ...

        # compute the MSE between the optimal solution and the corresponding y sample 
        # ...
        # ...

        # add MSE for the current sample to the total error
        total_err += MSE

        # append the optimal solution to the list of best solutions 
        best_sols.append(x)

    total_err *= 1/X.shape[0]

    return total_err, best_sols
```

3. Define `ray.remote` functions to return the solution associated to the best
   individual and the fitness of an individual. These functions must have the same
   arguments. The first argument is always the `Callable` that represents the tree of
   the individual. 
```python
@ray.remote
def eval_sols(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
              S: SimplicialComplex, u_0: C.CochainP0, penalty: dict) -> List[npt.NDArray]:

    _, best_sols = eval_MSE_sol(individual, X, y, S, u_0)

    return best_sols

@ray.remote
def eval_fitness(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
                 S: SimplicialComplex, u_0: C.CochainP0, penalty: dict) -> Tuple[float, ]:

    total_err, _ = eval_MSE_sol(individual, X, y, S, u_0)

    # add penalty on length of the tree to promote simpler solutions
    objval = total_err + penalty["reg_param"]*indlen

    return objval,
```

4. Define the main function to run the symbolic regression.
```python
def stgp_poisson():
    # read parameters from YAML file
    with open("ex1.yaml") as config_file:
        config_file_data = yaml.safe_load(config_file)

    # generate the mesh and build the corresponding SimplicialComplex object 
    # (using dctkit)
    # ...
    # ...

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

    # create the Symbolic Regression Problem object
    GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file_data)

    penalty = config_file_data["gp"]["penalty"]

    # store shared objects (such as the SimplicialComplex objects) common to the eval 
    # functions (fitness, best sols)
    GPprb.store_eval_common_params({'S': S, 'u_0': u_0, 'penalty': penalty})

    # define the parameters of the eval functions associated to the training, 
    # validation and test sets and store the corresponding data in the shared objs space
    param_names = ('X', 'y')
    datasets = {'train': [X_train, y_train], 'test': [X_train, y_train]}

    GPprb.store_eval_dataset_params(param_names, datasets)

    # register the functions for the (parallel) evaluation of the fitness and the best # individuals' solutions
    GPprb.register_eval_funcs(fitness=eval_fitness.remote, test_sols=eval_sols.remote)

    # register the map function and define features (such as individual length) that
    # must be passed to the eval functions
    feature_extractors = [len] 
    GPprb.register_map(feature_extractors)

    # run the symbolic regression problem
    GPprb.run(print_log=True, plot_best_individual_tree=False)

    # recover the solution associated to the best individual among all the populations
    u_best = GPprb.toolbox.map(GPprb.toolbox.evaluate_test_sols, (GPprb.best,))[0]

    ray.shutdown()
    
    # plot the solution
    # ...
    # ...
```

The complete notebook for this example can be found in the `examples` directory.