"""Deterministic Galerkin operators for the original radial-bin model.

The angular substitution s = |r13-r12| gives sin(theta)dtheta = s ds/(r12*r13).
Angular integrals are analytic. Only the two radial integrals need quadrature.
Because u_i(r12) u_l(r12) vanishes unless i=l, triple operators need N^3
entries per channel, rather than N^4. Channels are TTA, TPQ, triplet hopping,
and polaron Dexter hopping, respectively.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.special import roots_legendre

from .basis import RadialBasis
from .config import Numerics, Parameters

CACHE_VERSION = 1


@dataclass(frozen=True)
class Operators:
    basis: RadialBasis
    reaction: np.ndarray  # integral u_i^2 / r^6 dr; diagonal only
    rate: np.ndarray  # integral u_i / r^4 dr
    triple: np.ndarray  # (channel, r12 bin, r13 bin, r23 bin)
    diffusion: np.ndarray  # (channel, source bin, destination bin)
    metadata: dict

    def __post_init__(self):
        n = self.basis.size
        for name, shape in (
            ("reaction", (n,)),
            ("rate", (n,)),
            ("triple", (4, n, n, n)),
            ("diffusion", (2, n, n)),
        ):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                value.shape != shape
                or not np.all(np.isfinite(value))
                or np.any(value < 0)
            ):
                raise ValueError(f"invalid {name} operator")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _metadata(parameters, numerics, basis):
    return dict(
        version=CACHE_VERSION,
        edges=basis.edges.tolist(),
        reaction_cutoff=parameters.r0,
        diffusion_cutoff=parameters.r0_diffusion,
        dexter_length=parameters.dexter_length,
        quadrature_order=numerics.quadrature_order,
    )


def _quadrature(a, b, nodes, weights, split=None):
    edges = [a, b] if split is None or not a < split < b else [a, split, b]
    xs, ws = [], []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        xs.append(lower + (nodes + 1) * (upper - lower) / 2)
        ws.append(weights * (upper - lower) / 2)
    return np.concatenate(xs), np.concatenate(ws)


def build_operators(parameters: Parameters, numerics: Numerics) -> Operators:
    basis = RadialBasis.geometric(parameters.r0, numerics.r_max, numerics.bins)
    edges, widths, uniform = basis.edges, basis.widths, basis.uniform
    reaction = (edges[:-1] ** -5 - edges[1:] ** -5) / (5 * widths)
    rate = (edges[:-1] ** -3 - edges[1:] ** -3) / (3 * uniform)
    n = basis.size
    triple = np.zeros((4, n, n, n))
    diffusion = np.zeros((2, n, n))
    nodes, weights = roots_legendre(numerics.quadrature_order)
    cutoff, length = parameters.r0_diffusion, parameters.dexter_length
    for i in range(n):
        r, wr = _quadrature(edges[i], edges[i + 1], nodes, weights)
        r, wr = r[:, None], wr[:, None]
        for j in range(n):
            t, wt = _quadrature(edges[j], edges[j + 1], nodes, weights, cutoff)
            t, wt = t[None, :], wt[None, :]
            measure = wr * wt * t / r
            low = np.maximum(np.abs(r - t)[..., None], edges[:-1])
            high = np.minimum((r + t)[..., None], edges[1:])
            angular = np.maximum(high**2 - low**2, 0) / 2
            normalization = widths[i] * uniform[j] * uniform
            kernel = np.broadcast_to(t**-6, measure.shape)
            hop = np.where(t >= cutoff, t**-6, 0)
            dexter = np.exp(-np.maximum(t - cutoff, 0) / length) * (t >= cutoff)
            for channel, values in enumerate((kernel, kernel, hop, dexter)):
                triple[channel, i, j] = (
                    np.einsum("ab,abc->c", measure * values, angular) / normalization
                )

            if i == j:
                continue  # The conserving diagonal is set after physical scaling.
            lower = np.maximum(np.abs(r - t), max(parameters.r0, cutoff))
            upper = np.maximum(r + t, lower)
            inverse = (lower**-4 - upper**-4) / 4
            exponential = length * (
                (lower + length) * np.exp(-(lower - cutoff) / length)
                - (upper + length) * np.exp(-(upper - cutoff) / length)
            )
            for channel, angular_hop in enumerate((inverse, exponential)):
                diffusion[channel, j, i] = np.sum(measure * angular_hop) / (
                    uniform[i] * uniform[j]
                )

    return Operators(
        basis,
        reaction,
        rate,
        triple,
        np.maximum(diffusion, 0),
        _metadata(parameters, numerics, basis),
    )


def get_operators(
    parameters: Parameters, numerics: Numerics, cache_dir: str | Path | None = None
) -> Operators:
    """Read a validated numeric NPZ cache or compute it; never unpickle code."""
    if cache_dir is None:
        return build_operators(parameters, numerics)
    basis = RadialBasis.geometric(parameters.r0, numerics.r_max, numerics.bins)
    metadata = _metadata(parameters, numerics, basis)
    serialized = json.dumps(metadata, sort_keys=True)
    key = hashlib.sha256(serialized.encode()).hexdigest()
    directory = Path(cache_dir)
    path = directory / f"operators-{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["metadata"])) != metadata:
                raise ValueError(f"cache metadata mismatch: {path}")
            if not np.array_equal(data["edges"], basis.edges):
                raise ValueError(f"cache radial grid mismatch: {path}")
            return Operators(
                basis,
                data["reaction"],
                data["rate"],
                data["triple"],
                data["diffusion"],
                metadata,
            )
    operators = build_operators(parameters, numerics)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata=serialized,
                edges=basis.edges,
                reaction=operators.reaction,
                rate=operators.rate,
                triple=operators.triple,
                diffusion=operators.diffusion,
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return operators
