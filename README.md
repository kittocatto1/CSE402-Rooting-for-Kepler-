# Rooting for Kepler

CSE 402 (Numerical Methods) group project, BUET. We benchmark root finders
for Kepler's equation, `E - e sin E = M`: Newton-Raphson, Danby's quartic
method, Markley's non-iterative solver, and two with-memory methods (NWM9 and
NWM11) by Mittal et al. Each iterative solver is paired with four starting
guesses and run on a 7,400-point (e, M) grid. We measure convergence order,
cost, robustness and the effect on radial-velocity fits made with RadVel. We
also add a safeguarded NWM9/NWM11 that removes most of the published methods'
failures in the hard corner (e > 0.9, M < 0.1).

## Papers

| Paper | DOI / ID | Used for | PDF in `papers/` |
|---|---|---|---|
| Mittal, Panday & Jäntschi, *Mathematics* 12(22), 3490 (2024) | `10.3390/math12223490` | NWM9 | yes (CC BY) |
| Mittal, Panday, Jäntschi & Bolunduț, *AIMS Mathematics* 10(3), 5421-5443 (2025) | `10.3934/math.2025250` | NWM11 | yes (CC BY) |
| Danby, *Celestial Mechanics* 40, 303-312 (1987) | `10.1007/BF01235847` | Danby solver | no (Springer) |
| Markley, *Celest. Mech. Dyn. Astron.* 63, 101-111 (1995) | `10.1007/BF00691917` | Markley solver | no (Springer) |
| Napier, arXiv preprint (2024) | `arXiv:2411.15374` | Napier starting guess | yes (arXiv) |
| Fulton et al., *PASP* 130, 044504 (2018) | `10.1088/1538-3873/aaaaa8` | RadVel | yes (arXiv preprint) |
| Matthies et al., *Japan J. Indust. Appl. Math.* 33, 751-766 (2016) | `10.1007/s13160-016-0229-5` | base scheme of NWM9 | yes (arXiv preprint) |
| Solaiman & Hashim, *Intell. Autom. Soft Comput.* 27(2), 379-390 (2021) | `10.32604/iasc.2021.015285` | base scheme of NWM11 | yes (CC BY) |
| Kipping, *MNRAS Letters* 434, L51-L55 (2013) | `10.1093/mnrasl/slt075` | eccentricity sample in the grid | yes (arXiv preprint) |

The first six are the base papers from the proposal. `papers/README.md`
gives full citations, the licence of each file, the planet-system papers
(DOI only), and notes on transcribing the formulas.

## Repository layout

```
configs/      one YAML file per experiment, plus shared defaults
data/         RV datasets (git-ignored); data/README.md records the sources
docs/         the proposal and an earlier report draft
figures/      figures built by scripts/make_report_figures.py from results/
papers/       source papers we may redistribute; papers/README.md lists all
report/       the final report: main.tex (reads ../figures/) and the built A_01.pdf
results/      result tables written by the scripts; see results/README.md
scripts/      one entry point per experiment, plus the figure builder
src/keplerbench/
  core/         solver and guess interfaces, cost counters, registry
  solvers/      Newton, Danby, Markley, NWM9, NWM11, safeguarded variants
  guesses/      simple, canonical, RadVel and Napier starting guesses
  reference/    50-digit mpmath ground-truth roots
  evaluation/   metrics, convergence order, cost model, robustness, propagation
  experiments/  grid, sweep runner and the five experiments
  rv/           RV model, datasets and the RadVel bridge
  io/           config loading, result tables and provenance sidecars
  plotting/     figure functions (they plot tables, they never compute metrics)
tests/        pytest suite
```

## Installation

Python 3.10 or later.

```
python -m venv .venv
source .venv/bin/activate
pip install -e .            # core: numpy, scipy, pandas, matplotlib, mpmath, pyyaml
pip install -e ".[rv]"      # adds radvel and emcee, for the RV experiments
pip install -e ".[dev]"     # adds pytest and pytest-cov
```

`requirements.txt` lists the same core packages and pytest.

## Reproducing the results

Run the scripts from the repository root. Each script with a config takes an
optional config path; the default is the one shown.

| Experiment | Config | Script | Writes to `results/` |
|---|---|---|---|
| Verification against the papers | `configs/verification.yaml` | `scripts/run_verification.py` | `verification/order.csv`, `kepler_order.csv`, `summary.csv` |
| Grid benchmark (main experiment) | `configs/grid_benchmark.yaml` | `scripts/run_grid_benchmark.py` | `grid_benchmark/raw.csv`, `history.csv`, `timing.csv`, `summary.csv` |
| Safeguarded NWM9/NWM11 | `configs/safeguard.yaml` | `scripts/run_safeguard.py` | `safeguard/raw.csv`, `summary.csv`, `order.csv` |
| Solver tolerance to RV fit parameters | `configs/error_propagation.yaml` | `scripts/run_error_propagation.py` | `error_propagation/raw.csv`, `summary.csv`, `timing.csv`, `posterior_sigma.csv` |
| Measurement-noise Monte Carlo | `configs/monte_carlo.yaml` | `scripts/run_monte_carlo.py` | `monte_carlo/raw.csv`, `summary.csv` |
| Solver agreement on real RV data | none | `scripts/run_real_data_checks.py` | `real_data/summary.csv` |

The last three need the `rv` extra. The first run also fills
`results/reference_roots.csv`, the cache of ground-truth roots that every
error metric uses. `scripts/run_grid_benchmark.py --limit N` runs only the
first N grid points, for a quick smoke test.

When the experiments are done, rebuild every figure from the tables:

```
python scripts/make_report_figures.py          # rebuild all figures
python scripts/make_report_figures.py --list   # show what it would build
```

The figures go to `figures/`, as PDF and PNG, and are committed so they can
be viewed without running anything. Two PNGs over 1 MB are not committed;
their PDF versions are. The report in `report/` includes these figures.

`results/README.md` explains each table. Most tables have a `.meta.json`
sidecar with the git commit and a config fingerprint. The two large `raw.csv`
files (grid benchmark and safeguard) are not committed; the scripts above
regenerate them.

`keplerbench list` shows the registered solvers and guesses, and
`keplerbench solve --solver nwm9 --guess napier --e 0.9 --M 0.05` prints one
solve with its iteration trace.

## Tests

```
pytest
```

`tests/test_config.py` loads every config in `configs/`.
`tests/test_rv.py` uses RadVel, so it needs the `rv` extra. Without it, run
`pytest --ignore=tests/test_rv.py`.

## Data

The RV datasets are RadVel's example files and are not committed.
`data/README.md` gives the download link and references for each:
`k2-24.csv` (K2-24, 32 HIRES points), `hd164922.txt` (HD 164922, 401 HIRES
points from 3 instrument eras) and `k2-131.txt` (K2-131, 70 HARPS-N and PFS
points). The error-propagation and Monte Carlo studies use the HD 164922
epochs and uncertainties with an injected orbit (e = 0.35).

## Results at a glance

Every number below is copied from the file named next to it.

Verification (`results/verification/summary.csv`). Each solver passed all 8
paper test functions and the Kepler correctness check, with zero wrong roots.

| Solver | Claimed order | Measured order, mean (min-max) |
|---|---|---|
| Newton | 2 | 2.0 |
| Danby | 4 | 4.0 |
| NWM9 | 8.8989 | 8.926 (8.885-9.000) |
| NWM11 | 10.7446 | 10.761 (10.715-10.834) |

Grid benchmark (`results/grid_benchmark/summary.csv`), 7,400 points per
solver and guess:

- Markley converged on all 7,400 points, with a weighted cost of 3.34 per
  solve.
- Danby converged on all 7,400 points with the canonical, Napier and RadVel
  guesses, and on 7,398 with the simple guess.
- NWM9 converged on 7,400 points with the Napier guess, but on only 7,358
  with the simple guess.

Safeguard (`results/safeguard/summary.csv`). There are 29,600 solves per
solver (7,400 points x 4 guesses). Failures in total, and in the hard corner
(e > 0.9, M < 0.1):

| Solver | Published | Memory off | Guarded |
|---|---|---|---|
| NWM9 | 81 (65) | 18 (13) | 17 (13) |
| NWM11 | 74 (74) | 9 (9) | 3 (3) |
| Danby, for reference | 2 (2) | | |

## Team

| Member | ID | Owns |
|---|---|---|
| Anisa Binte Asad | 2105036 | Integration and framework: `core/`, `io/`, `reference/`, `evaluation/metrics.py` and `aggregate.py`, `plotting/style.py`, `cli.py`, `scripts/make_report_figures.py` |
| Dipit Saha | 2105050 | Classical solvers (`solvers/newton.py`, `danby.py`, `markley.py`), the cost model (`evaluation/cost_model.py`), cost figures |
| Atika Tabassum Suchi | 2105053 | With-memory solvers (`solvers/nwm9.py`, `nwm11.py`, `_withmemory_base.py`), convergence order, the verification experiment, convergence figures |
| S. M. A. M. Mahdi | 2105056 | Starting guesses (`guesses/`), the grid and sweep runner, the grid benchmark, robustness metrics, the safeguard (`solvers/safeguarded.py`, `experiments/safeguard.py`), grid figures |
| Fariha Ifrat | 2105059 | RadVel downstream study: `rv/`, `experiments/error_propagation.py` and `monte_carlo.py`, `evaluation/propagation.py`, propagation figures |

Each module's docstring names its owner. The `rv/` and propagation modules
have no owner line in the code. Their owner is given in `configs/` and
`tests/test_rv.py`.

## Papers and licences

The journal PDFs in `papers/` are CC BY 4.0. The arXiv preprints are
included only as course reference copies, and each one's arXiv abs page is
linked in `papers/README.md` as the authoritative source. Danby (1987),
Markley (1995) and the planet-system papers are under publisher copyright and
are not included. The RV data belongs to the RadVel project and the original
survey teams. Cite them as listed in `data/README.md` and `papers/README.md`.
