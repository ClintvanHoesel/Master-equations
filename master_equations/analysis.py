"""Apparent annihilation coefficients inferred from transient luminescence."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .results import Result


@dataclass(frozen=True)
class AnnihilationEstimate:
    """Legacy kTT2 (yield), kTT1 (half-yield time), and their ratio.

    These use dn/dt = -K_RD*n - k*n²/2. With TPQ present they are apparent
    quenching coefficients, not independently identified TTA rates.
    """

    from_yield: float
    from_half_time: float
    half_time: float

    @property
    def ratio(self):
        return (
            self.from_yield / self.from_half_time if self.from_half_time > 0 else None
        )


def estimate_annihilation(result: Result) -> AnnihilationEstimate:
    """Fit the thesis' yield estimators with bracketed roots and interpolation.

    Retains the thesis tail approximation: remaining excitons eventually decay
    radiatively. Extend the simulation until this correction is negligible.
    """
    p = result.parameters
    if p.generation_rate != 0 or p.initial_density <= 0:
        raise ValueError(
            "yield estimates require zero generation and positive initial density"
        )
    total_yield = float(result.losses[-1, 0] + result.density[-1])
    fraction = total_yield / p.initial_density
    if not 0 < fraction <= 1 + 1e-8:
        raise ValueError("radiative yield must be in (0, initial density]")
    half_yield = total_yield / 2
    if result.losses[-1, 0] < half_yield:
        raise ValueError(
            "half of the radiative yield has not been observed; increase t_end"
        )
    cumulative, unique = np.unique(np.r_[0, result.losses[:, 0]], return_index=True)
    times = np.r_[0, result.time][unique]
    half_time = float(PchipInterpolator(cumulative, times)(half_yield))
    if fraction >= 1 - 1e-10:
        return AnnihilationEstimate(0.0, 0.0, half_time)

    def equation(x):
        return (1.0 if x == 0 else np.log1p(x) / x) - fraction

    upper = 1.0
    while equation(upper) > 0:
        upper *= 2
        if not np.isfinite(upper):
            raise ValueError(
                "yield is too small to resolve an annihilation coefficient"
            )
    x = brentq(equation, 0, upper, xtol=1e-14)
    elapsed = p.decay_rate * half_time
    numerator = 2 * np.exp(-elapsed) - 1
    if numerator < 0:
        raise ValueError("half-yield time exceeds pure decay; refine time sampling")
    half_x = numerator / np.expm1(-elapsed) ** 2
    scale = 2 * p.decay_rate / p.initial_density
    return AnnihilationEstimate(float(scale * x), float(scale * half_x), half_time)
