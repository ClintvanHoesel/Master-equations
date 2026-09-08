"""Small geometry helpers retained for exploratory notebooks."""

import warnings

import numpy as np


def cubic_diffusion_cutoff(lattice_spacing, guest_fraction, *, discrete=True):
    if (
        not np.isfinite(lattice_spacing)
        or lattice_spacing <= 0
        or not 0 < guest_fraction <= 1
    ):
        raise ValueError(
            "require positive lattice_spacing and guest_fraction in (0, 1]"
        )
    if discrete:
        thresholds = (
            0.3116,
            0.137356,
            0.097644,
            0.080117,
            0.046181,
            0.033705,
            0.02908,
            0.021869,
            0.018406,
        )
        radii = (
            0.792934,
            1.20368,
            1.66843,
            1.9053,
            2.01517,
            2.35707,
            2.71738,
            2.82742,
            3.07427,
        )
        for threshold, radius in zip(thresholds, radii, strict=True):
            if guest_fraction > threshold:
                return lattice_spacing * radius
        warnings.warn(
            "guest fraction is below the tabulated range; using continuum cutoff",
            UserWarning,
            stacklevel=2,
        )
    return 0.792934 * lattice_spacing * guest_fraction ** (-1 / 3)


def squared_separation(r1, theta1, phi1, r2, theta2, phi2):
    r1, theta1, phi1, r2, theta2, phi2 = np.broadcast_arrays(
        r1, theta1, phi1, r2, theta2, phi2
    )
    cosine = np.cos(theta1) * np.cos(theta2) + np.cos(phi1 - phi2) * np.sin(
        theta1
    ) * np.sin(theta2)
    return np.maximum(r1**2 + r2**2 - 2 * r1 * r2 * cosine, 0)


def triangle_angle(r12, r13, r23):
    r12, r13, r23 = np.broadcast_arrays(r12, r13, r23)
    if np.any(r12 <= 0) or np.any(r13 <= 0) or np.any(r23 < 0):
        raise ValueError("r12 and r13 must be positive, and r23 nonnegative")
    if not all(np.all(np.isfinite(r)) for r in (r12, r13, r23)):
        raise ValueError("distances must be finite")
    cosine = (r12**2 + r13**2 - r23**2) / (2 * r12 * r13)
    return np.arccos(np.clip(cosine, -1, 1))


def inverse_sixth(distance, cutoff):
    """Hard cutoff avoids 0/0 at excluded distances."""
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    distance = np.asarray(distance, dtype=float)
    if not np.all(np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("distance must be finite and nonnegative")
    return np.divide(
        1.0, distance**6, out=np.zeros_like(distance), where=distance >= cutoff
    )
