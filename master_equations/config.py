"""Validated physical parameters. Lengths are nm and times are microseconds."""

from dataclasses import asdict, dataclass
from math import isfinite, pi


@dataclass(frozen=True)
class Parameters:
    """Inputs retain the rate conventions of the original thesis scripts.

    ``dexter_rate`` is the dimensionless hopping strength at the diffusion
    cutoff; its physical rate is ``dexter_rate / lifetime``.
    Generation acts on unoccupied sites. Polarons have a fixed density.
    """

    lifetime: float = 1.0
    lattice_spacing: float = 1.0
    guest_fraction: float = 1.0
    initial_fraction: float = 0.02
    polaron_density: float = 0.001
    generation_rate: float = 0.0
    tta_radius: float = 3.5
    tpq_radius: float = 3.5
    diffusion_radius: float = 0.0
    dexter_rate: float = 0.0
    dexter_length: float = 0.15
    reaction_cutoff: float | None = None
    diffusion_cutoff: float | None = None
    effective_diffusion_lattice: bool = True
    effective_reaction_lattice: bool = False
    closure: str = "kirkwood"

    def __post_init__(self):
        positive = ("lifetime", "lattice_spacing", "dexter_length")
        nonnegative = (
            "polaron_density",
            "generation_rate",
            "tta_radius",
            "tpq_radius",
            "diffusion_radius",
            "dexter_rate",
        )
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not isfinite(value) or value < 0 or (name in positive and value == 0):
                bound = "positive" if name in positive else "nonnegative"
                raise ValueError(f"{name} must be finite and {bound}")
        for name in ("reaction_cutoff", "diffusion_cutoff"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.guest_fraction <= 1:
            raise ValueError("guest_fraction must be in (0, 1]")
        if not 0 <= self.initial_fraction <= 1:
            raise ValueError("initial_fraction must be in [0, 1]")
        if self.polaron_density > self.site_density:
            raise ValueError("polaron_density exceeds site_density")
        if self.initial_density + self.polaron_density > self.site_density * (
            1 + 1e-14
        ):
            raise ValueError("initial excitons and polarons exceed the available sites")
        if self.closure not in ("kirkwood", "pair"):
            raise ValueError("closure must be 'kirkwood' or 'pair'")
        for name in ("effective_diffusion_lattice", "effective_reaction_lattice"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

    @property
    def site_density(self):
        return self.guest_fraction / self.lattice_spacing**3

    @property
    def initial_density(self):
        return self.initial_fraction * self.site_density

    @property
    def decay_rate(self):
        return 1 / self.lifetime

    @property
    def base_cutoff(self):
        return self.lattice_spacing * (4 * pi / (3 * 8.40192150285141)) ** (1 / 3)

    @property
    def r0(self):
        if self.reaction_cutoff is not None:
            return self.reaction_cutoff
        return self.base_cutoff * (
            self.guest_fraction ** (-1 / 3) if self.effective_reaction_lattice else 1
        )

    @property
    def r0_diffusion(self):
        if self.diffusion_cutoff is not None:
            return self.diffusion_cutoff
        return self.base_cutoff * (
            self.guest_fraction ** (-1 / 3) if self.effective_diffusion_lattice else 1
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Numerics:
    """The final bin [r_max, 2*r_max) is a fixed far-field reservoir."""

    bins: int = 24
    r_max: float = 60.0
    quadrature_order: int = 32
    method: str = "Radau"
    rtol: float = 1e-6
    atol: float = 1e-9

    def __post_init__(self):
        if type(self.bins) is not int or self.bins < 2:
            raise ValueError("bins must be an integer >= 2")
        if type(self.quadrature_order) is not int or self.quadrature_order < 4:
            raise ValueError("quadrature_order must be an integer >= 4")
        for name in ("r_max", "rtol", "atol"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.method not in ("Radau", "BDF", "LSODA", "RK45", "DOP853"):
            raise ValueError("unsupported integration method")

    def to_dict(self):
        return asdict(self)
