"""The safeguard experiment: does guarding the memory fix the corner failures?

Runs NWM9 and NWM11 three ways on the full benchmark grid - as published,
with the memory switched off, and with the safeguard from
``solvers/safeguarded.py`` - next to Danby as the reference for robustness.
Then re-measures convergence order for the guarded solvers, to show the
safeguard does not cost the order the papers prove.

Writes to results/safeguard/:

  raw.csv      one row per solve, same columns as the grid benchmark
  summary.csv  failures, failure rates by region and mean iterations
  order.csv    measured order: paper test functions and Kepler at M = 0.3

Owner: Mahdi.
"""

from __future__ import annotations

import statistics

import pandas as pd

import keplerbench.guesses  # noqa: F401  (registers the guesses)
import keplerbench.solvers  # noqa: F401  (registers the solvers)
import keplerbench.solvers.safeguarded  # noqa: F401
from keplerbench.evaluation.robustness import region
from keplerbench.experiments.grid import build_grid
from keplerbench.experiments.runner import run_sweep
from keplerbench.experiments.verification import (
    NWM9_TEST_FUNCTIONS,
    PAPER_TEST_FUNCTIONS,
    measure_order_on_kepler,
    verify_order_on_test_functions,
)
from keplerbench.io.config import load_config
from keplerbench.io.results_io import _write_meta, results_path, save_results

EXPERIMENT = "safeguard"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per solver: failures, regional failure rates, mean iterations."""
    df = df.assign(region=region(df), converged=df["converged"].astype(bool))
    rows = []
    for solver, g in df.groupby("solver", sort=False):
        corner = g[g.region == "hard_corner"]
        rest = g[g.region != "hard_corner"]
        ok = g[g.converged]
        row = {
            "solver": solver,
            "n_solves": len(g),
            "n_failed": int((~g.converged).sum()),
            "n_failed_corner": int((~corner.converged).sum()),
            "fail_pct_corner": 100 * (~corner.converged).mean(),
            "fail_pct_rest": 100 * (~rest.converged).mean(),
            "mean_iterations": ok.iterations.mean(),
        }
        for guess, gg in g.groupby("guess"):
            row[f"n_failed_{guess}"] = int((~gg.converged).sum())
            row[f"mean_iterations_{guess}"] = gg[gg.converged].iterations.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _order_rows(points) -> list[dict]:
    rows = []
    pairs = [("nwm9", NWM9_TEST_FUNCTIONS), ("nwm9_guarded", NWM9_TEST_FUNCTIONS),
             ("nwm11", PAPER_TEST_FUNCTIONS), ("nwm11_guarded", PAPER_TEST_FUNCTIONS)]
    for name, functions in pairs:
        tf = verify_order_on_test_functions(name, n_iterations=8, functions=functions)
        orders = [r["measured_order"] for r in tf]
        rows.append({"solver": name, "check": "paper_test_functions",
                     "e": None, "measured_order": statistics.mean(orders),
                     "min": min(orders), "max": max(orders),
                     "passed": sum(bool(r["passed"]) for r in tf)})
        for r in measure_order_on_kepler(name, points, n_iterations=10):
            order = r["measured_order"]
            rows.append({"solver": name, "check": "kepler_M0.3", "e": r.get("e"),
                         "measured_order": None if order is None else float(order),
                         "min": None, "max": None, "passed": None})
    return rows


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    points = build_grid(cfg.grid)
    print(f"safeguard: {len(points)} points, {len(cfg.solvers)} solvers, "
          f"{len(cfg.guesses)} guesses")

    from keplerbench.reference.mpmath_reference import reference_roots_grid
    reference_roots = reference_roots_grid([e for e, _ in points], [M for _, M in points])
    results = run_sweep(cfg.solvers, cfg.guesses, points, tol=cfg.tol,
                        max_iter=cfg.max_iter, reference_roots=reference_roots)
    raw_path = save_results(results, EXPERIMENT, "raw.csv", config=cfg)
    print(f"  wrote {raw_path}")

    summary = summarise(pd.read_csv(raw_path))
    path = results_path(EXPERIMENT, "summary.csv")
    summary.to_csv(path, index=False)
    _write_meta(path, EXPERIMENT, len(summary), cfg)
    print(summary[["solver", "n_failed", "n_failed_corner", "fail_pct_corner",
                   "mean_iterations"]].round(3).to_string(index=False))

    order_points = [(float(e), 0.3) for e in cfg.extra.get("order_eccentricities", [])]
    order = pd.DataFrame(_order_rows(order_points))
    path = results_path(EXPERIMENT, "order.csv")
    order.to_csv(path, index=False)
    _write_meta(path, EXPERIMENT, len(order), cfg)
    print(f"  wrote {path}")
