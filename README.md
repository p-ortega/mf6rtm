# MF6RTM: Reactive Transport Model via the MODFLOW 6 and PHREEQCRM APIs
![Tests](https://github.com/p-ortega/mf6rtm/actions/workflows/tests_main.yml/badge.svg)
![Tests](https://github.com/p-ortega/mf6rtm/actions/workflows/tests_macos.yml/badge.svg)

[![GitHub tag](https://img.shields.io/github/tag/mf6rtm/mf6rtm.svg)](https://github.com/p-ortega/mf6rtm/tags/latest)
[![PyPI License](https://img.shields.io/pypi/l/mf6rtm)](https://pypi.python.org/pypi/mf6rtm)
<!-- [![PyPI Status](https://img.shields.io/pypi/status/modflowapi.png)](https://pypi.python.org/pypi/mf6rtm) -->
<!-- [![PyPI Format](https://img.shields.io/pypi/format/modflowapi)](https://pypi.python.org/pypi/mf6rtm) -->
[![PyPI Version](https://img.shields.io/pypi/v/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![PyPI Versions](https://img.shields.io/pypi/pyversions/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)

## Benchmarks
Benchmark comparing model results against PHT3D are in `benchmark/`. Each folder contains a Jupyter notebook to write and execute an MF6RTM model via the MUP3D class. Additionally, PHT3D files are provided in the corresponding `pht3d` directory for each example.

## Considerations
The current version is intended to work with structured grids (dis object in MF6) and one MF6 simulation that includes the flow and transport solutions. No support is currently provided for a 'flow then transport scheme,' meaning that advanced packages cannot be incorporated yet.

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
The package is still not published in PyPi but it can be installed through pip by cloning the repository. 

Once the repository is already cloned in your local machine, please just type the following in your conda prompt:

```commandline
conda env create -f env.yml
```

Otherwise, the package can be installed through pip as:

```commandline
pip install git+https://github.com/p-ortega/mf6rtm.git#egg=mf6rtm
```

## Funding
The developing of mf6rtm was kindly funded and supported by [Intera, Inc](https://www.intera.com).

## Authors
Pablo Ortega (Portega)