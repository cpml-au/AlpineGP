[![Documentation Status](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/latest/?badge=latest)

# AlpineGP

_AlpineGP_ is a Python library that helps to build algorithms that can identify _symbolic_ models
of _physical systems_ starting from data. It performs **symbolic regression** using a
_strongly-typed genetic programming_ approach implemented in the [`DEAP`](https://github.com/alucantonio/DEAP)
library. As a natural language for expressing physical models, it leverages the
**discrete calculus** framework
defined and implemented in the library [`dctkit`](https://github.com/alucantonio/dctkit).

_AlpineGP_ has been introduced in the paper [_Discovering interpretable physical models
with symbolic regression and discrete exterior calculus_](https://iopscience.iop.org/article/10.1088/2632-2153/ad1af2),
along with several benchmark problems.


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

1. Define the function that computes the prediction associated to an _individual_ (model expression tree).
Its arguments are a _function_ obtained by parsing the individual tree and possibly other
parameters (datasets to compare the individual with). It returns both an _error metric_ between
the prediction and the data and the prediction itself. 
```python
def eval_MSE_sol(individual: Callable, D: Dataset):

    # ...
    return MSE, prediction
```

2. Define the functions that return the **prediction** and the **fitness** 
   associated to an individual. These functions **must** have the same
   arguments. The first argument is **always** the `Callable` that represents the
   individual tree. The functions **must** be decorated with `ray.remote` to support
   distributed evaluation (multiprocessing).
```python
@ray.remote
def predict(individual: Callable, indlen: int, D: Dataset, penalty: float) -> float:

    _, pred = eval_MSE_sol(individual, D)

    return pred

@ray.remote
def fitness(individual: Callable, length: int, D: Dataset, penalty: float) -> Tuple[float, ]:

    MSE, _ = eval_MSE_sol(individual, D)

    # add penalty on length of the tree to promote simpler solutions
    fitness = MSE + penalty*length

    # return value MUST be a tuple
    return fitness,
```

3. Set and solve the symbolic regression problem.
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

# define extra common arguments of fitness and predict functions
common_params = {'penalty': penalty}

# create the Symbolic Regression Problem object
gpsr = gps.GPSymbolicRegressor(pset=pset, fitness=fitness.remote,
                               predict_func=predict.remote, common_data=common_params,
                               feature_extractors=[len],
                               print_log=True, 
                               config_file_data=config_file_data)

# define training Dataset object (to be used for model fitting)
train_data = Dataset("D", X_train, y_train)

# solve the symbolic regression problem
gpsr.fit(train_data)

# recover the solution associated to the best individual among all the populations
u_best = gpsr.predict(train_data)

# plot the solution
# ...
# ...
```

A complete example notebook can be found in the `examples` directory.

## Citing
```
@article{Manti_2024,
    doi = {10.1088/2632-2153/ad1af2},
    url = {https://dx.doi.org/10.1088/2632-2153/ad1af2},
    year = {2024},
    publisher = {IOP Publishing},
    volume = {5},
    number = {1},
    pages = {015005},
    author = {Simone Manti and Alessandro Lucantonio},
    title = {Discovering interpretable physical models using symbolic regression and discrete exterior calculus},
    journal = {Machine Learning: Science and Technology}
}
```