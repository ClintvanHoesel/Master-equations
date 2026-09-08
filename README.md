# Master equations

A Python implementation of the triplet–polaron pair-correlation model developed
for my Master thesis at TU Eindhoven. It simulates triplet
decay, triplet–triplet annihilation (TTA), triplet–polaron quenching (TPQ), Förster
triplet hopping, Dexter polaron hopping, and continuous exciton generation.

The project has explicit configuration, deterministic operator generation,
transient and steady-state solvers, parameter sweeps, portable results, and
scientific figures. The original equations and their far-field boundary condition
are retained. See [the model notes](docs/model.md) for conventions and numerical
changes, and [migration notes](docs/migration.md) for the old variable names.

## Running and installing the Master Equation solver

The package can be installed and run as:

```bash
python -m pip install --no-build-isolation -e '.[plot,excel,notebook,dev]'
python -m master_equations transient --config examples/transient.toml --plot --excel
```

This produces `results/transient.npz`, `.csv`, `.json`, `.xlsx`, `.png`, and `.svg`.
The NPZ stores the complete trajectory and configuration. The JSON records units,
parameters, numerical settings, and a summary. CSV contains population, losses,
and effective rates; Excel also contains correlations and metadata.

Every command can instead be prefixed, to run with conda, with `conda run -n me`, where `me` is the conda environment used. For example:

```bash
conda run -n me python -m master_equations steady --config examples/steady.toml --plot
conda run -n me python -m master_equations sweep --config examples/sweep.toml
conda run -n me python -m master_equations plot results/transient.npz --correlations --output results/correlations.svg
```

## Configure an experiment

Copy a file from `examples/` and edit its TOML tables:

```toml
[parameters]
initial_fraction = 0.02
polaron_density = 0.001
tta_radius = 3.5
tpq_radius = 3.5
dexter_rate = 30.0
generation_rate = 0.0

[numerics]
bins = 24
r_max = 60.0
quadrature_order = 32
rtol = 1e-6
atol = 1e-9

[solver]
t_end = 10.0
samples = 161
```

Lengths are **nm**, times **μs**, densities **nm⁻³**, generation/decay rates
**μs⁻¹**, and effective reaction coefficients **nm³/μs**. `dexter_rate` retains the
thesis convention: the hopping rate at the cutoff is `dexter_rate / lifetime`.

The last radial bin is a fixed reservoir from `r_max` to `2*r_max`; `bins` includes
it. Increase the bin count, domain, and quadrature order independently when checking
convergence for research results. The default settings are an example, not a
convergence guarantee for every parameter regime.

Transient commands default to a 10-lifetime decay. `steady` defaults to generation
0.01 μs⁻¹ unless specified in the configuration. For steady runs, the `[solver]`
table accepts `max_time`, `rtol`, and `atol` for convergence of the derivatives;
`[numerics]` tolerances control ODE integration. A failed convergence check raises
an error instead of reporting an equilibrium. With generation, use a positive
`initial_fraction`: normalized correlations are undefined at exactly zero density.

```bash
conda run -n me python -m master_equations transient --config examples/transient.toml --bins 48 --quadrature-order 64 --output results/refined
conda run -n me python -m master_equations precompute --config examples/transient.toml
conda run -n me python -m master_equations --help
```

Operators are cached in `.cache/master-equations/` using full-precision geometry,
quadrature settings, and a format version. Reaction strengths and generation do
not invalidate geometry caches. Use `--no-cache` to compute in memory. Results
written to the same output prefix replace that prefix's previous files.


## License

[PolyForm Noncommercial 1.0.0](LICENSE.md).
