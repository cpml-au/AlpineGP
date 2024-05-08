[![Documentation Status](https://readthedocs.org/projects/alpine/badge/?version=latest)](https://alpine.readthedocs.io/en/latest/?badge=latest)

# AlpineGP

_AlpineGP_ is a Python library that solves **symbolic regression** problems using
_Genetic Programming_. It provides a high-level interface to the
[`DEAP`](https://github.com/alucantonio/DEAP) library and leverages the high-performance
distributed computing functionalities provided by the [`ray`](https://www.ray.io) library. 

Beside solving classical symbolic regression problems involving algebraic equations
(see, for example, the benchmark problems contained in the
[SRBench](https://github.com/cavalab/srbench)
repository), _AlpineGP_ is specifically design to help identifying _interpretable_, 
_symbolic_ models of _physical systems_ starting from data. To this aim, it exploits as a natural and
effective language to express physical models (i.e., conservation laws) a **discrete
calculus** framework, including tools from discrete differential geometry and discrete
exterior calculus, defined and implemented in the library
[`dctkit`](https://github.com/alucantonio/dctkit).

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

1. Define the function that computes the prediction associated to an _individual_
(model expression tree). Its arguments may be a _function_ obtained by parsing the
individual tree and possibly other parameters, such as the dataset needed to evaluate
the model. It returns both an _error metric_ between the prediction and the data and
the prediction itself. 
```python
def eval_MSE_sol(individual, dataset):

    # ...
    return MSE, prediction
```

1. Define the functions that return the **prediction** and the **fitness** 
   associated to an individual. These functions **must** have the same
   arguments. In particular:
   - the first argument is **always** the batch of trees to be evaluated by the
     current worker;
   - the second argument **must** be the `toolbox` object used to compile the 
     individual trees into callable functions;
   - the third argument **must** be the dataset needed for the evaluation of the
     individuals.
   Both functions **must** be decorated with `ray.remote` to support
   distributed evaluation (multiprocessing).
```python
@ray.remote
def predict(trees, toolbox, data):

    callables = compile_individuals(toolbox, trees)

    preds = [None]*len(trees)

    for i, ind in enumerate(callables):
        _, preds[i] = eval_MSE_sol(ind, data)

    return preds

@ray.remote
def fitness(trees, toolbox, true_data):
    callables = compile_individuals(toolbox, trees)

    fitnesses = [None]*len(trees)

    for i, ind in enumerate(callables):
        MSE, _ = eval_MSE_sol(ind, data)
        
        # each fitness MUST be a tuple (required by DEAP)
        fitnesses[i] = (MSE,)

    return fitnesses
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
                               print_log=True, 
                               config_file_data=config_file_data)

# wrap tensors corresponding to train and test data into Dataset objects (to be passed to
# fit and predict methods)
train_data = Dataset("D", X_train, y_train)
test_data = Dataset("D", X_test, y_test)

# solve the symbolic regression problem
gpsr.fit(train_data)

# compute the prediction on the test dataset given by the best model found during the SR
pred_test = gpsr.predict(test_data)
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