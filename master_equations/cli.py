"""Command-line experiments configured with readable TOML files."""

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

from .config import Numerics, Parameters
from .model import MasterEquation
from .operators import get_operators
from .results import Result
from .solvers import IntegrationError, solve_steady_state, solve_transient
from .sweep import sweep


def _config(path):
    if path is None:
        return {}
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    unknown = set(config) - {"parameters", "numerics", "solver", "sweep"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    if any(not isinstance(value, dict) for value in config.values()):
        raise ValueError("configuration sections must be TOML tables")
    return config


def _export(result, prefix, excel=False, plot=False):
    prefix = Path(prefix)
    result.save(str(prefix) + ".npz")
    result.to_csv(str(prefix) + ".csv")
    Path(str(prefix) + ".json").write_text(
        json.dumps(dict(**result.metadata, summary=result.summary()), indent=2) + "\n",
        encoding="utf-8",
    )
    if excel:
        result.to_excel(str(prefix) + ".xlsx")
    if plot:
        from .plotting import plot_overview, save_figure

        figure = plot_overview(result)
        save_figure(figure, str(prefix) + ".svg")
        save_figure(figure, str(prefix) + ".png")


def _parser():
    parser = argparse.ArgumentParser(
        description="Triplet/polaron pair-correlation master equations"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (
        ("transient", "Integrate a decay or driven transient"),
        ("steady", "Find a steady state and verify its residual"),
        ("precompute", "Build and cache radial operators"),
        ("sweep", "Run a Cartesian parameter sweep"),
    ):
        sub = commands.add_parser(command, help=description)
        sub.add_argument("--config", type=Path, help="TOML experiment configuration")
        sub.add_argument(
            "--cache-dir", type=Path, default=Path(".cache/master-equations")
        )
        sub.add_argument("--no-cache", action="store_true")
        sub.add_argument("--bins", type=int)
        sub.add_argument("--r-max", type=float)
        sub.add_argument("--quadrature-order", type=int)
        sub.add_argument("--generation-rate", type=float)
        if command == "precompute":
            continue
        sub.add_argument(
            "--output", type=Path, help="Output prefix (directory for sweeps)"
        )
        sub.add_argument("--excel", action="store_true")
        sub.add_argument("--plot", action="store_true")
        if command in ("transient", "sweep"):
            sub.add_argument("--t-end", type=float)
            sub.add_argument("--samples", type=int)
        if command in ("steady", "sweep"):
            sub.add_argument("--max-time", type=float)
        if command == "sweep":
            sub.add_argument(
                "--mode", choices=("transient", "steady"), default="transient"
            )
            sub.add_argument(
                "--axis",
                action="append",
                default=[],
                metavar="NAME=V1,V2",
                help="Numeric parameter axis; repeat for a Cartesian product",
            )
    sub = commands.add_parser("plot", help="Plot a saved NPZ result")
    sub.add_argument("result", type=Path)
    sub.add_argument("--output", type=Path, default=Path("results/overview.svg"))
    sub.add_argument("--correlations", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plot":
            from .plotting import plot_correlations, plot_overview, save_figure

            result = Result.load(args.result)
            plot = plot_correlations if args.correlations else plot_overview
            save_figure(plot(result), args.output)
            print(f"Saved {args.output}")
            return 0
        config = _config(args.config)
        physics = config.get("parameters", {}).copy()
        numerics = config.get("numerics", {}).copy()
        for name in ("bins", "r_max", "quadrature_order"):
            if getattr(args, name) is not None:
                numerics[name] = getattr(args, name)
        if args.generation_rate is not None:
            physics["generation_rate"] = args.generation_rate
        elif args.command == "steady" and "generation_rate" not in physics:
            physics["generation_rate"] = 0.01
        parameters, numerics = Parameters(**physics), Numerics(**numerics)
        cache_dir = None if args.no_cache else args.cache_dir
        if args.command == "precompute":
            operators = get_operators(parameters, numerics, cache_dir)
            print(
                json.dumps(
                    dict(bins=operators.basis.size, metadata=operators.metadata),
                    indent=2,
                )
            )
            return 0
        mode = args.mode if args.command == "sweep" else args.command
        options = config.get("solver", {}).copy()
        allowed = (
            {"t_end", "samples"}
            if mode == "transient"
            else {"max_time", "rtol", "atol"}
        )
        for name in ("t_end", "samples", "max_time"):
            if getattr(args, name, None) is not None:
                options[name] = getattr(args, name)
        if set(options) - allowed:
            raise ValueError(
                f"invalid {mode} solver options: {sorted(set(options) - allowed)}"
            )
        output = args.output or Path("results") / args.command
        if args.command == "sweep":
            axes = config.get("sweep", {}).copy()
            for axis in args.axis:
                name, raw = axis.split("=", 1)
                axes[name] = [float(value) for value in raw.split(",")]
            output.mkdir(parents=True, exist_ok=True)
            with (output / "summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = None
                for index, point in enumerate(
                    sweep(
                        parameters,
                        axes,
                        numerics=numerics,
                        mode=mode,
                        cache_dir=cache_dir,
                        **options,
                    )
                ):
                    prefix = output / f"run-{index:04d}"
                    _export(point.result, prefix, args.excel, args.plot)
                    row = dict(
                        run=prefix.name, **point.values, **point.result.summary()
                    )
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    print(json.dumps(row))
        else:
            model = MasterEquation(parameters, numerics, cache_dir)
            solver = solve_transient if mode == "transient" else solve_steady_state
            result = solver(model, **options)
            _export(result, output, args.excel, args.plot)
            print(json.dumps(result.summary(), indent=2))
        return 0
    except (
        ValueError,
        TypeError,
        OSError,
        KeyError,
        IntegrationError,
        ImportError,
    ) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
