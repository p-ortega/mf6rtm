"""
The MF6RTM (Modflow 6 Reactive Transport Model) package is a Python package
for reactive transport modeling via the MODFLOW 6 and PhreeqcRM APIs.
"""

import sys
import types
import warnings

warnings.filterwarnings("ignore", message="builtin type.*has no __module__", category=DeprecationWarning)

__author__ = "Pablo Ortega"
from . import config, mup3d, simulation, utils
from ._version import __version__
from .io import yaml_reader

# Optionally, expose base from mup3d
from .mup3d import base
from .simulation.mf6api import Mf6API
from .simulation.phreeqcbmi import PhreeqcBMI
from .simulation.solver import DT_FMT, Mf6RTM, run_cmd, solve, time_units_dict


def _deprecated_module(old_name, new_module):
    proxy = types.ModuleType(old_name)

    def __getattr__(name):
        # The import system probes dunders like __path__; don't warn for those
        if name.startswith("__"):
            raise AttributeError(name)
        warnings.warn(
            f"{old_name} is deprecated, import from {new_module.__name__} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(new_module, name)

    proxy.__getattr__ = __getattr__
    sys.modules[old_name] = proxy


# Old paths from when config/ and utils/ were folders; kept so scripts and pickles still load.
_deprecated_module(__name__ + ".config.config", config)
_deprecated_module(__name__ + ".config.yaml_reader", yaml_reader)
_deprecated_module(__name__ + ".utils.utils", utils)

# Define public API
__all__ = [
    "__version__",
    "mup3d",
    "simulation",
    "config",
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
