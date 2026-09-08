"""Transient integration and residual-checked steady states."""

import numpy as np
from scipy.integrate import solve_ivp

from .model import MasterEquation
from .results import Result


class IntegrationError(RuntimeError):
    """The numerical solver failed or produced a nonphysical state."""


class ConvergenceError(IntegrationError):
    """Steady-state integration reached its time limit before convergence."""


def _integrate(model, span, initial, times=None):
    p, options = model.parameters, model.numerics
    if p.generation_rate > 0 and initial[3] <= 0:
        raise ValueError(
            "use initial_fraction > 0 with generation: "
            "normalized g is undefined at zero density"
        )
    solution = solve_ivp(
        model.rhs,
        span,
        initial,
        t_eval=times,
        method=options.method,
        rtol=options.rtol,
        atol=options.atol,
    )
    if not solution.success:
        raise IntegrationError(solution.message)
    if not np.all(np.isfinite(solution.y)):
        raise IntegrationError("integration produced non-finite values")
    tolerance = 100 * options.atol
    if (
        np.min(solution.y[3]) < -tolerance
        or np.max(solution.y[3]) > p.site_density - p.polaron_density + tolerance
    ):
        raise IntegrationError("density left the physically allowed interval")
    if np.min(solution.y[4:] / np.tile(model.basis.uniform, 2)[:, None]) < -tolerance:
        raise IntegrationError(
            "negative pair correlations; refine the radial grid and quadrature"
        )
    return solution


def solve_transient(
    model: MasterEquation | None = None,
    times=None,
    *,
    t_end: float | None = None,
    samples: int = 161,
) -> Result:
    model = model or MasterEquation()
    if times is None:
        end = 10 * model.parameters.lifetime if t_end is None else t_end
        if not np.isfinite(end) or end <= 0 or type(samples) is not int or samples < 2:
            raise ValueError(
                "t_end must be positive and samples must be an integer >= 2"
            )
        times = np.r_[0.0, np.geomspace(end * 1e-8, end, samples - 1)]
    elif t_end is not None:
        raise ValueError("supply times or t_end, not both")
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
        or times[-1] <= 0
    ):
        raise ValueError(
            "times must be a strictly increasing nonnegative array of length >= 2"
        )
    sol = _integrate(model, (0, times[-1]), model.initial_state(), times)
    return Result(
        sol.t,
        sol.y.T,
        model.mean_field(sol.t),
        model.parameters,
        model.numerics,
        model.basis,
        sol.nfev,
    )


def solve_steady_state(
    model: MasterEquation | None = None, *, max_time=None, rtol=1e-6, atol=1e-9
) -> Result:
    """Require every density/correlation derivative to be small per lifetime.

    Integrated losses are excluded: they continue growing at equilibrium.
    A normalized residual <= 1 passes the requested absolute/relative criterion.
    A time limit alone never labels a trajectory as steady.
    """
    model = model or MasterEquation()
    lifetime = model.parameters.lifetime
    max_time = 1000 * lifetime if max_time is None else max_time
    if any(not np.isfinite(value) or value <= 0 for value in (max_time, rtol, atol)):
        raise ValueError("max_time, rtol and atol must be finite and positive")
    state = model.initial_state()
    times, states = [0.0], [state.copy()]
    time, step, evaluations = 0.0, lifetime, 0
    scales = np.r_[
        model.parameters.site_density, model.basis.uniform, model.basis.uniform
    ]
    while time < max_time:
        end = min(time + step, max_time)
        sol = _integrate(model, (time, end), state)
        times.extend(sol.t[1:].tolist())
        states.extend(sol.y.T[1:].copy())
        evaluations += sol.nfev
        time, state = end, sol.y[:, -1]
        residual = float(
            np.max(
                np.abs(model.rhs(time, state)[3:])
                * lifetime
                / (atol * scales + rtol * np.abs(state[3:]))
            )
        )
        if residual <= 1:
            times = np.asarray(times)
            return Result(
                times,
                np.asarray(states),
                model.mean_field(times),
                model.parameters,
                model.numerics,
                model.basis,
                evaluations,
                True,
                residual,
                dict(max_time=max_time, rtol=rtol, atol=atol),
            )
        step *= 2
    raise ConvergenceError(
        f"steady state did not converge by {max_time:g} us "
        f"(normalized residual {residual:.3g} > 1)"
    )
