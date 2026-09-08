"""Master equations for triplet excitons and polarons, after C. van Hoesel."""

from .analysis import AnnihilationEstimate, estimate_annihilation
from .basis import RadialBasis
from .config import Numerics, Parameters
from .model import MasterEquation
from .results import Result
from .solvers import (
    ConvergenceError,
    IntegrationError,
    solve_steady_state,
    solve_transient,
)
from .sweep import sweep

__all__ = [
    "AnnihilationEstimate",
    "ConvergenceError",
    "IntegrationError",
    "MasterEquation",
    "Numerics",
    "Parameters",
    "RadialBasis",
    "Result",
    "solve_steady_state",
    "solve_transient",
    "sweep",
    "estimate_annihilation",
]
__version__ = "1.0.0"
