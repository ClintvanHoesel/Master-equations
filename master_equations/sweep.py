"""Cartesian parameter sweeps with explicit inputs and reusable geometry caches."""

from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path

from .config import Numerics, Parameters
from .model import MasterEquation
from .results import Result
from .solvers import solve_steady_state, solve_transient


@dataclass(frozen=True)
class SweepPoint:
    values: dict
    result: Result


def sweep(
    parameters: Parameters,
    axes: dict,
    *,
    numerics: Numerics | None = None,
    mode="transient",
    cache_dir: str | Path | None = None,
    **solver_options,
):
    """Yield each run in axis insertion order; inputs are never mutated."""
    if mode not in ("transient", "steady"):
        raise ValueError("mode must be 'transient' or 'steady'")
    if not axes:
        raise ValueError("provide at least one sweep axis")
    allowed = {field.name for field in fields(Parameters)}
    if set(axes) - allowed:
        raise ValueError(f"unknown parameters: {sorted(set(axes) - allowed)}")
    values = [tuple(value) for value in axes.values()]
    if any(not value for value in values):
        raise ValueError("sweep axes cannot be empty")
    solver = solve_transient if mode == "transient" else solve_steady_state
    for combination in product(*values):
        point = dict(zip(axes, combination, strict=True))
        model = MasterEquation(replace(parameters, **point), numerics, cache_dir)
        yield SweepPoint(point, solver(model, **solver_options))
