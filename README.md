[![Documentation Status](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/latest/?badge=latest)

# AlpineGP

_AlpineGP_ is a Python library that helps to build algorithms that can identify _symbolic_ models
of _physical systems_ starting from data. It performs **symbolic regression** using a
_strongly-typed genetic programming_ approach implemented in the [`DEAP`](https://github.com/alucantonio/DEAP)
library. As a natural language for expressing physical models, it leverages the
**discrete calculus** framework
defined and implemented in the library [`dctkit`](https://github.com/alucantonio/dctkit).


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

Setting up a symbolic regression problem in _AlpineGP_ involves several key steps:

1. Define the function to compute the prediction associated to an _individual_ (model expression tree).
It takes as inputs _Callable_ obtained by parsing the individual tree and possibly other
parameters (datasets to compare the individual with). It returns an _error metric_ between
the prediction and the ground truth and the prediction itself. 
```python
def eval_MSE_sol(individual: Callable, X: npt.NDArray, y: npt.NDArray) -> Tuple[float, float]:

    # ...
    return MSE, prediction
```

2. Define `ray.remote` wrapper functions that return the prediction associated to the best
   individual and the **fitness** of an individual. These functions **must** have the same
   arguments. The first argument is always the `Callable` that represents the tree of
   the individual. 
```python
@ray.remote
def eval_sols(individual: Callable, indlen: int, X: npt.NDArray, y: npt.NDArray,
              penalty: float) -> float:

    _, pred = eval_MSE_sol(individual, X, y)

    return pred

@ray.remote
def eval_fitness(individual: Callable, length: int, X: npt.NDArray, y: npt.NDArray,
                 penalty: float) -> Tuple[float, ]:

    MSE, _ = eval_MSE_sol(individual, X, y)

    # add penalty on length of the tree to promote simpler solutions
    fitness = MSE + penalty*length

    # return value MUST be a tuple
    return fitness,
```

3. Define the main function to run the symbolic regression.
```python
# read parameters from YAML file
with open("ex1.yaml") as config_file:
    config_file_data = yaml.safe_load(config_file)

# ...
# ...

# load datasets...

# define the primitive set (input/output types)
pset = gp.PrimitiveSetTyped(...)

# rename arguments of the tree function
pset.renameArguments(ARG0="u")

# create the Symbolic Regression Problem object
GPprb = gps.GPSymbRegProblem(pset=pset, config_file_data=config_file_data)

# store shared objects (such as the SimplicialComplex objects) common to the eval 
# functions (fitness, best sols)
GPprb.store_eval_common_params({'S': S, 'u_0': u_0, 'penalty': penalty})

# define the parameters of the eval functions associated to the training, 
# validation and test sets and store the corresponding data in the shared objs space
param_names = ('X', 'y')
datasets = {'train': [X_train, y_train], 'test': [X_train, y_train]}

GPprb.store_eval_dataset_params(param_names, datasets)

# register the functions for the (parallel) evaluation of the fitness and the 
# prediction associated to the test set (if present)
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

A complete example notebook can be found in the `examples` directory.