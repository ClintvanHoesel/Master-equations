"""Coupled density and normalized pair-correlation equations.

State: [integrated decay, integrated TTA, integrated TPQ, n_T, alpha_TT,
alpha_TP]. The polaron density and g_PP=1 remain fixed. The last radial
coefficient is pinned at its initial far-field value, as in the thesis.
"""

from pathlib import Path

import numpy as np

from .config import Numerics, Parameters
from .operators import get_operators


class MasterEquation:
    def __init__(
        self,
        parameters: Parameters | None = None,
        numerics: Numerics | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.parameters = parameters or Parameters()
        self.numerics = numerics or Numerics()
        p = self.parameters
        self.operators = get_operators(p, self.numerics, cache_dir)
        self.basis = self.operators.basis
        reaction_scales = p.decay_rate * np.array([p.tta_radius**6, p.tpq_radius**6])
        hop_scales = p.decay_rate * np.array([p.diffusion_radius**6, p.dexter_rate])
        self.rate = 4 * np.pi * reaction_scales[:, None] * self.operators.rate
        self.reaction = reaction_scales[:, None] * self.operators.reaction
        self.triple = (
            2
            * np.pi
            * np.r_[reaction_scales, hop_scales][:, None, None, None]
            * self.operators.triple
        )
        self.diffusion = (
            2
            * np.pi
            * p.site_density
            * hop_scales[:, None, None]
            * self.operators.diffusion
        )
        uniform = self.basis.uniform
        for matrix in self.diffusion:
            np.fill_diagonal(matrix, -(uniform @ matrix) / uniform)
        if p.closure == "pair":
            self.triple.fill(0)

    def initial_state(self):
        return np.r_[
            np.zeros(3),
            self.parameters.initial_density,
            self.basis.uniform,
            self.basis.uniform,
        ]

    def contract(self, channel, pair12, pair23, pair13):
        """Apply the compact tensor in the thesis' original dot-product order."""
        return pair12 * np.einsum("ijk,j,k->i", self.triple[channel], pair13, pair23)

    def rhs(self, time, state):
        p = self.parameters
        n = state[3]
        bins = self.basis.size
        tt, tp = state[4 : 4 + bins], state[4 + bins :]
        pp = self.basis.uniform
        polaron = p.polaron_density
        ktt, ktp = self.rate[0] @ tt, self.rate[1] @ tp
        losses = np.array([p.decay_rate * n, n**2 * ktt, n * polaron * ktp])
        density = p.generation_rate * (p.site_density - polaron - n) - losses.sum()

        dtt = -2 * self.reaction[0] * tt + 2 * (tt @ self.diffusion[0])
        dtp = -self.reaction[1] * tp + tp @ (self.diffusion[0] + self.diffusion[1])
        if p.generation_rate:
            if n <= 0:
                raise ValueError(
                    "normalized correlations require positive density "
                    "when generation is enabled"
                )
            dtt += (
                2
                * p.generation_rate
                / n
                * (p.site_density * pp + (polaron - p.site_density) * tt - polaron * tp)
            )
            dtp += (
                p.generation_rate
                / n
                * (p.site_density * pp + (polaron - p.site_density) * tp - polaron * pp)
            )

        if p.closure == "kirkwood":
            contract = self.contract
            dtt += -2 * n * contract(0, tt, tt, tt) - 2 * polaron * contract(
                1, tt, tp, tp
            )
            dtp += -n * contract(0, tp, tp, tt) - polaron * contract(1, tp, pp, tp)
            total_rate = n * ktt + polaron * ktp
            dtt += 2 * tt * total_rate
            dtp += tp * total_rate
            dtt += 2 * polaron * (contract(2, tt, tp, tp) - contract(2, tp, tt, tp))
            dtp += polaron * (contract(2, tp, pp, tp) - contract(2, pp, tp, tp))
            dtp += n * (contract(3, tp, tt, tp) - contract(3, tt, tp, tp))

        dtt[-1] = dtp[-1] = 0
        return np.r_[losses, density, dtt, dtp]

    def mean_field(self, times):
        """Stable analytic solution of dn/dt = source - linear*n - quadratic*n².

        Includes polarons and the same finite radial domain as the pair model.
        Handles zero generation and zero TTA without complex arithmetic.
        """
        p = self.parameters
        times = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must be finite and nonnegative")
        quadratic, tpq = self.rate @ self.basis.uniform
        linear = p.decay_rate + p.generation_rate + p.polaron_density * tpq
        source = p.generation_rate * (p.site_density - p.polaron_density)
        discriminant = np.sqrt(linear**2 + 4 * quadratic * source)
        equilibrium = 2 * source / (linear + discriminant)
        delta = p.initial_density - equilibrium
        decay = np.exp(-discriminant * times)
        return equilibrium + delta * decay / (
            1 + quadratic * delta / discriminant * (-np.expm1(-discriminant * times))
        )
