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
Genetic Programming approach to symbolic regression via the
[`DEAP`](https://github.com/alucantonio/DEAP) library.

## Installation

Clone the git repository and install the required libraries listed in the file
`requirements.txt`. Then, launch the following command

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


