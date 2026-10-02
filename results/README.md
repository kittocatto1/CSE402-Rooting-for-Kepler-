# Results

Every file here is written by a script in `scripts/`. Nothing is edited by
hand. Only the small tables are committed, so that the numbers in the README
and the report can be checked without rerunning anything. The large
per-solve dumps are ignored by git and come back when you rerun the scripts:

| Ignored file | Size | Rebuilt by |
|---|---|---|
| `grid_benchmark/raw.csv` | about 17 MB, 125,800 rows | `scripts/run_grid_benchmark.py` |
| `safeguard/raw.csv` | about 29 MB, 207,200 rows | `scripts/run_safeguard.py` |

Every derived table can be rebuilt from these two files plus the committed
ones.

## Committed tables

`reference_roots.csv` holds the cached ground-truth roots `E(e, M)` for the
7,400 grid points. Columns are `e, M, E, dps`. Each root is found with mpmath
bracketing at 50 digits (`reference/mpmath_reference.py`) and written as a
double. Every error metric is measured against this file.

`verification/` (from `run_verification.py`, config `verification.yaml`)

- `order.csv`: one row per solver and test function from the source papers.
  It gives the measured order of convergence (at 2000 digits), the claimed
  order, and whether they agree.
- `kepler_order.csv`: the measured order on Kepler's equation itself, at
  M = 0.3, for e from 0.1 to 0.999.
- `summary.csv`: one row per solver. It combines the order check and a
  correctness check on a set of Kepler points (4,975 in the current run;
  `kepler_check_points` in the config), reporting the fraction converged,
  the wrong-root count and the maximum absolute error. `passed` is the
  verification gate.

`grid_benchmark/` (from `run_grid_benchmark.py`, config `grid_benchmark.yaml`)

- `summary.csv`: one row per (solver, guess) over the 7,400-point grid. It
  gives convergence counts, iterations, weighted cost (cost model in
  `evaluation/cost_model.py`), median wall-clock time and correct digits.
- `timing.csv`: wall-clock seconds per solve on a 50-point subset, the median
  of 1000 repeats each.
- `history.csv`: per-iteration records (iterate, residual, error, cost
  counters) on the small history sub-grid named in the config. The
  residual-history figure (`figures/convergence/residual_histories`) is
  drawn from it. The two order figures in the report read `verification/`.

`safeguard/` (from `run_safeguard.py`, config `safeguard.yaml`)

- `summary.csv`: one row per solver (Danby; NWM9 and NWM11 as published,
  memoryless and guarded). It gives failures in total, in the hard corner
  (e > 0.9, M < 0.1) and per guess, plus the mean iterations of the
  converged solves. Each solver has 29,600 solves (7,400 points x 4 guesses).
- `order.csv`: the measured order of the published and guarded methods.
  `paper_test_functions` rows give the mean, minimum and maximum over the
  paper's 8 test functions, and `passed` is the number of those functions
  that passed. `kepler_M0.3` rows give the order on Kepler at M = 0.3. An
  empty value means there were too few usable error terms for an estimate.

`error_propagation/` (from `run_error_propagation.py`, config
`error_propagation.yaml`, needs the `rv` extra)

- `raw.csv`: one maximum-likelihood fit per (solver, solver tolerance) on the
  HD 164922 epochs with an injected orbit. It includes the fitted parameters,
  the number of Kepler solves and the fit time.
- `summary.csv`: the shift of each fitted parameter from the fit at the
  reference tolerance (1e-14), in the parameter's own units, relative to its
  value, and in units of the posterior sigma. All three are signed.
- `posterior_sigma.csv`: posterior widths from one MCMC run.
- `timing.csv`: median fit time per solver, compared with RadVel's native
  solver.

`monte_carlo/` (from `run_monte_carlo.py`, config `monte_carlo.yaml`, needs
the `rv` extra)

- `raw.csv`: one fit per noise realisation.
- `summary.csv`: the spread of each fitted parameter that measurement noise
  causes. This spread is the scale for the solver-induced shifts above.

`real_data/` (from `run_real_data_checks.py`, no config file, needs the `rv`
extra)

- `summary.csv`: one fit per (dataset, solver) of the real, unmodified
  velocities of K2-24, HD 164922 and K2-131, at solver tolerance 1e-14. It
  checks that the five solvers agree on real data. It is not an orbit
  measurement, since K2-24 and HD 164922 are multi-planet systems fitted with
  one planet.

## Provenance sidecars

Most tables have a `<table>.meta.json` next to them. It records the git
commit, whether the working tree was dirty, the config name, a fingerprint of
the config, the row count and the write time. The verification, safeguard
and real-data tables and the grid-benchmark `raw` and `history` tables have
one (the real-data sidecar has no config fields, as that script takes no
config). The grid-benchmark
`summary.csv` and `timing.csv`, the `error_propagation/` and `monte_carlo/`
tables, and `reference_roots.csv` do not, because their writers do not
produce one yet.
