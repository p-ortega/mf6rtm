# MF6RTM: Reactive Transport Model via the MODFLOW 6 and PHREEQCRM APIs
![Tests](https://github.com/p-ortega/mf6rtm/actions/workflows/tests_main.yml/badge.svg)
![Tests](https://github.com/p-ortega/mf6rtm/actions/workflows/tests_macos.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/p-ortega/mf6rtm/badge.svg?branch=develop)](https://coveralls.io/github/p-ortega/mf6rtm?branch=main)
[![PyPI License](https://img.shields.io/pypi/l/mf6rtm)](https://pypi.python.org/pypi/mf6rtm)
<!-- [![PyPI Status](https://img.shields.io/pypi/status/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm) -->
<!-- [![PyPI Format](https://img.shields.io/pypi/format/mf6rtm)](https://pypi.python.org/pypi/mf6rtm) -->
[![PyPI Version](https://img.shields.io/pypi/v/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![PyPI Versions](https://img.shields.io/pypi/pyversions/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![DOI](https://zenodo.org/badge/798559356.svg)](https://doi.org/10.5281/zenodo.17118951)

## Benchmarks
Benchmark comparing model results against PHT3D are in `benchmark/`. Each folder contains a Jupyter notebook to write and execute an MF6RTM model via the MUP3D class. Additionally, PHT3D files are provided in the corresponding `pht3d` directory for each example.

## Considerations
The current version is intended to work with structured grids (dis object in MF6), unstructured grids (disv) and one MF6 simulation that includes the flow and transport solutions. No support is currently provided for a 'flow then transport scheme,' meaning that advanced packages cannot be incorporated yet.

On the PHREEQC side, the following have been included:

- Solution
- Equilibrium phases
- Cation Exchange
- Surface Complexation
- Kinetic Phases

Most options for each phreeqc block can be passed by adding list with options. However, not all options had been tested, so please create an issue if any option is not working or crashing the model.

## Software requirements
All dependencies and executables are included in this repo. This package extensively uses [modflowapi](https://github.com/MODFLOW-USGS/modflowapi) and [phreeqcrm](https://github.com/usgs-coupled/phreeqcrm)

## Installation

### Quick Start with pip

The package can be installed via pip:

```commandline
pip install mf6rtm
```

### Manual Installation with Conda/Mamba

If you prefer conda/mamba, create a dedicated environment:

```commandline
mamba env create -f env.yml
mamba activate mf6rtm-dev
```

After activating the environment, install the MODFLOW6 executables:

```commandline
pip install modflow-devtools
get-modflow --subset mf6,libmf6,gridgen :python
```

Once installed, the executables in `envs/[env-name]/bin` will be automatically invoked whenever mf6rtm runs within the environment.

### Custom MODFLOW Versions

If you need custom or older versions of mf6 (e.g., for running PESTPP on an HPC cluster), place them in a separate directory and use the provided utility to bring them to the model working directory:

```Python
from mf6rtm import utils

utils.prep_bins(model_dir, src_path=path_to_bins)
```
### Running the benchmark notebooks
We have provided some benchmarks in the form of Jupyter notebooks. We have also included the executables needed to run them out of the box. Nevertheless, they can also be run using the executables downloaded with modflow-devtools.

## Developing

### With Pixi (Recommended)

For development, we recommend using pixi for fast, reproducible environments:

```commandline
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/mf6rtm.git
cd mf6rtm

# Install development environment with all dependencies
pixi install

# Run tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Run linting
pixi run lint

# Test with specific Python version
pixi run -e py311 test

```

### With Conda/Mamba

Alternatively, use conda/mamba with the provided environment file:

```commandline
# Install environment
conda env create -f env.yml
conda activate mf6rtm-dev

# Install development dependencies
pip install -r requirements_dev.txt
```

The development dependencies for testing are located in `requirements_dev.txt`. We have also provided dependencies with flopy and pyemu inside the repo but feel free to use your own distribution.

## Funding
The developing of mf6rtm was kindly funded and supported by [Intera, Inc](https://www.intera.com).

## Authors
Pablo Ortega (Portega)
