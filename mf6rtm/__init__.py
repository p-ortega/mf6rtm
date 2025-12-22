"""
The MF6RTM (Modflow 6 Reactive Transport Model) package is a Python package
for reactive transport modeling via the MODFLOW 6 and PhreeqcRM APIs.
"""

__author__ = "Pablo Ortega"
__version__ = "0.2.1+develop"  #

from . import mup3d
from . import simulation
from .utils import utils

from .simulation.solver import run_cmd, solve
from .simulation.mf6api import Mf6API
from .simulation.phreeqcbmi import PhreeqcBMI
from .simulation.solver import Mf6RTM

# Optionally, expose base from mup3d
from .mup3d import base

# Define public API
__all__ = [
    "mup3d",
    "simulation",
    "utils",
    "Mf6API",
    "PhreeqcBMI",
    "Mf6RTM",
    "run_cmd",
    "Solver",
    "solve",
    "DT_FMT",
    "time_units_dict",
    "base",
]
