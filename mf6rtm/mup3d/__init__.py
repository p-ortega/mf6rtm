"""
The mup3d module provides function to write input files for phreeqcrm and adds
functionality to write files externally in a modflow-like fashion.

History: The Model Utility Preprocessor 3D (MUP3D) was envisioned by Mr. Beaker
as a successor to PHT3D.
"""

from .base import *
from .base import __getattr__ as __getattr__

__all__ = ["Mup3d"]
