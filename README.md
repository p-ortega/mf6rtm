
# MODFLOW 6 and PHREEQC Reactive Transport Modeling

<img align="left" height="200" alt="mf6rtm logo" src="https://raw.githubusercontent.com/p-ortega/mf6rtm/main/mf6rtm/assets/mf6rtm.png">

**Main branch:** [![CI Linux](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi.yml/badge.svg?branch=main)](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi.yml) [![CI macOS](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi-macos.yml/badge.svg?branch=main)](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi-macos.yml) [![Docs](https://readthedocs.org/projects/mf6rtm/badge/?version=stable)](https://mf6rtm.readthedocs.io/en/stable/)
[![Coverage Status](https://coveralls.io/repos/github/p-ortega/mf6rtm/badge.svg?branch=main)](https://coveralls.io/github/p-ortega/mf6rtm?branch=main)

**Develop branch:** [![CI Linux](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi.yml/badge.svg?branch=develop)](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi.yml) [![CI macOS](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi-macos.yml/badge.svg?branch=develop)](https://github.com/p-ortega/mf6rtm/actions/workflows/ci-pixi-macos.yml) [![Docs](https://readthedocs.org/projects/mf6rtm/badge/?version=latest)](https://mf6rtm.readthedocs.io/en/latest/) [![Coverage Status](https://coveralls.io/repos/github/p-ortega/mf6rtm/badge.svg?branch=develop)](https://coveralls.io/github/p-ortega/mf6rtm?branch=develop)

[![PyPI License](https://img.shields.io/pypi/l/mf6rtm)](https://pypi.python.org/pypi/mf6rtm)
[![PyPI Version](https://img.shields.io/pypi/v/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![PyPI Versions](https://img.shields.io/pypi/pyversions/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17118951.svg)](https://doi.org/10.5281/zenodo.17118951)
![Platforms](https://img.shields.io/badge/platforms-linux%20%7C%20macOS%20%7C%20windows-blue)

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

### From conda-forge (coming soon)

A conda-forge recipe is being prepared but not yet submitted -- `conda install -c conda-forge mf6rtm` isn't available yet. Use the pip install above in the meantime.

### Installing MODFLOW 6 executables

After installing mf6rtm, install the MODFLOW 6 executables:

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

### With Pixi

For development, we use pixi for fast, reproducible environments:

```commandline
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/mf6rtm.git
cd mf6rtm

# Install development environment with all dependencies, including an
# editable install of mf6rtm itself -- `import mf6rtm` works right away,
# no separate `pip install -e .` step needed
pixi install

# Run tests -- this fetches the MODFLOW 6 executables it needs
# automatically (via the install-modflow/install-benchmark-bins task
# dependencies); run those two tasks directly if you just want the
# executables on disk for interactive use, without running the suite
pixi run test

# Run tests with coverage
pixi run test-cov

# Run linting
pixi run lint

# Test with a specific Python version (py311, py312, or py313)
pixi run -e py311 test
```
## Documentation

The documentation is available in [mf6rtm - read the docs](https://mf6rtm.readthedocs.io)

## Funding
The developing of mf6rtm was kindly supported by [INTERA, Inc](https://www.intera.com).