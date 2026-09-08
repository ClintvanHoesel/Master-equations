"""Orthonormal, piecewise constant radial basis with measure dr."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, eq=False)
class RadialBasis:
    edges: NDArray[np.float64]

    def __post_init__(self):
        edges = np.array(self.edges, dtype=float, copy=True)
        if (
            edges.ndim != 1
            or edges.size < 3
            or not np.all(np.isfinite(edges))
            or edges[0] <= 0
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError(
                "edges must contain >= 3 positive, increasing finite values"
            )
        edges.setflags(write=False)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def geometric(cls, r0: float, r_max: float, bins: int):
        if not 0 < r0 < r_max or type(bins) is not int or bins < 2:
            raise ValueError("require 0 < r0 < r_max and integer bins >= 2")
        return cls(np.r_[np.geomspace(r0, r_max, bins), 2 * r_max])

    @property
    def size(self):
        return self.edges.size - 1

    @property
    def widths(self):
        return np.diff(self.edges)

    @property
    def centers(self):
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def uniform(self):
        """Coefficients representing g(r)=1 inside the radial domain."""
        return np.sqrt(self.widths)

    def evaluate(self, radii: ArrayLike):
        """Return shape (bins, *radii.shape); support is half-open [a,b)."""
        radii = np.asarray(radii, dtype=float)
        shape = (self.size,) + (1,) * radii.ndim
        lower = self.edges[:-1].reshape(shape)
        upper = self.edges[1:].reshape(shape)
        return ((radii >= lower) & (radii < upper)) / self.uniform.reshape(shape)

    def reconstruct(self, coefficients: ArrayLike, radii: ArrayLike):
        return np.tensordot(coefficients, self.evaluate(radii), axes=([-1], [0]))
