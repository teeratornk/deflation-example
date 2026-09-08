"""Reference-space deflation examples; importing this module does not import Torch."""

from .problems import build_problem, reference_modes
from .solvers import LinearResult, deflated_cg, pdas

__version__ = "0.1.1"
__all__ = ["LinearResult", "build_problem", "reference_modes", "deflated_cg", "pdas"]
