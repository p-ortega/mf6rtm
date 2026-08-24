"""
The MF6RTM (Modflow 6 Reactive Transport Model) package is a Python package
for reactive transport modeling via the MODFLOW 6 and PhreeqcRM APIs.
"""

import warnings

warnings.filterwarnings("ignore", message="builtin type.*has no __module__", category=DeprecationWarning)

__author__ = "Pablo Ortega"
from . import mup3d, simulation
from ._version import __version__

# Optionally, expose base from mup3d
from .mup3d import base
from .simulation.mf6api import Mf6API
from .simulation.phreeqcbmi import PhreeqcBMI
from .simulation.solver import DT_FMT, Mf6RTM, run_cmd, solve, time_units_dict
from .utils import utils

# Define public API
__all__ = [
    "__version__",
    "mup3d",
    "simulation",
    "utils",
    "Mf6API",
    "PhreeqcBMI",
    "Mf6RTM",
    "run_cmd",
    "solve",
    "DT_FMT",
    "time_units_dict",
    "base",
]
