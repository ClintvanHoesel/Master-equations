"""Self-contained scientific figures without global styles or a LaTeX install."""

from pathlib import Path

import numpy as np

from .results import Result

COLORS = ("#2364aa", "#cf5c36", "#39856d")


def _figure(rows=2, columns=2, size=(10, 7)):
    from matplotlib.figure import Figure

    figure = Figure(figsize=size, layout="constrained", facecolor="white")
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", labelsize=9)
        axis.grid(alpha=0.16)
    return figure, axes


def plot_overview(result: Result):
    figure, axes = _figure()
    figure.suptitle("Triplet and polaron master equations", fontsize=15)
    time = result.time / result.parameters.lifetime
    axis = axes[0, 0]
    axis.plot(time, result.density, color=COLORS[0], label="Pair correlations")
    axis.plot(
        time, result.mean_field, color="#707780", linestyle="--", label="Mean field"
    )
    if np.any(result.density > 0):
        axis.set_yscale("log")
    axis.set(
        xlabel=r"Time $t/\tau_{RD}$",
        ylabel=r"Triplet density (nm$^{-3}$)",
        title="Population",
    )
    axis.legend(frameon=False)
    axis = axes[0, 1]
    for values, label, color in zip(
        result.losses.T, ("Radiative", "TTA", "TPQ"), COLORS, strict=True
    ):
        axis.plot(time, values, label=label, color=color)
    axis.set(
        xlabel=r"Time $t/\tau_{RD}$",
        ylabel=r"Lost density (nm$^{-3}$)",
        title="Cumulative losses",
    )
    axis.legend(frameon=False)
    axis = axes[1, 0]
    for values, label, color in zip(
        result.effective_rates.T, ("TTA", "TPQ"), COLORS[:2], strict=True
    ):
        axis.plot(time, values, label=label, color=color)
    if np.any(result.effective_rates > 0):
        axis.set_yscale("log")
    axis.set(
        xlabel=r"Time $t/\tau_{RD}$",
        ylabel=r"Rate coefficient (nm$^3$/$\mu$s)",
        title="Effective reaction rates",
    )
    axis.legend(frameon=False)
    axis = axes[1, 1]
    axis.stairs(
        result.g_tt[-1],
        result.basis.edges,
        color=COLORS[0],
        label=r"$g_{TT}$",
        baseline=None,
    )
    axis.stairs(
        result.g_tp[-1],
        result.basis.edges,
        color=COLORS[1],
        label=r"$g_{TP}$",
        baseline=None,
    )
    axis.axhline(1, color="#707780", linewidth=0.8, linestyle=":")
    axis.set(
        xscale="log",
        xlabel="Separation (nm)",
        ylabel="Normalized pair correlation",
        title=f"Correlations at {result.time[-1]:.3g} μs",
    )
    axis.legend(frameon=False)
    return figure


def plot_correlations(result: Result):
    figure, axes = _figure(1, 2, size=(10, 4))
    positive = result.time > 0
    if positive.sum() < 2:
        raise ValueError("correlation maps need at least two positive sample times")
    for axis, values, label in zip(
        axes.flat, (result.g_tt, result.g_tp), ("TT", "TP"), strict=True
    ):
        mesh = axis.pcolormesh(
            result.time[positive],
            result.basis.centers,
            values[positive].T,
            shading="nearest",
            cmap="viridis",
            vmin=0,
            vmax=max(1, float(values.max())),
        )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Time (μs)",
            ylabel="Separation (nm)",
            title=f"{label} pair correlation",
        )
        axis.grid(False)
        figure.colorbar(mesh, ax=axis, label=f"g{label}")
    return figure


def save_figure(figure, path: str | Path, *, dpi=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor="white")
    return path
