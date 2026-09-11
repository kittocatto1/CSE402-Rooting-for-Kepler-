"""Tests for the pipeline and the grid. Owner: Mahdi."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.grid import build_grid, pathological_grid, uniform_grid
from keplerbench.experiments.runner import run_sweep, solve_one


def test_solve_one_labels_the_guess():
    r = solve_one(get_solver("newton"), get_guess("simple"), e=0.2, M=1.0)
    assert r.guess == "simple"
    assert r.solver == "newton"


def test_guess_cost_is_not_charged_to_the_iteration():
    """The guess layer must be measured separately from the iteration, so
    the counters start at zero when the first step runs."""
    r = solve_one(get_solver("newton"), get_guess("simple"),
                  e=0.5, M=1.0, tol=0.0, max_iter=1, record_history=True)
    # iteration 0 record is the guess itself: one sin-only residual check.
    assert r.history[0].cost["sincos_pairs"] == 0


def test_uniform_grid_shape():
    with skip_if_unimplemented():
        pts = uniform_grid(n_e=5, n_M=4)
        assert len(pts) == 20
        assert all(0.0 <= e < 1.0 and 0.0 <= M <= math.pi for e, M in pts)


def test_pathological_grid_is_actually_in_the_corner():
    with skip_if_unimplemented():
        pts = pathological_grid()
        assert all(e >= 0.9 for e, _ in pts)
        assert all(M <= 0.1 for _, M in pts)


def test_build_grid_dispatch():
    with skip_if_unimplemented():
        pts = build_grid({"type": "uniform", "uniform": {"n_e": 3, "n_M": 3}})
        assert len(pts) == 9


def test_run_sweep_covers_every_combination():
    with skip_if_unimplemented():
        results = run_sweep(["newton"], ["simple"], [(0.1, 1.0), (0.5, 2.0)])
        assert len(results) == 2


# ----------------------------------------------------------------------
# Robustness metrics (evaluation/robustness.py). Owner: Mahdi.
# ----------------------------------------------------------------------
def _fake_results():
    import pandas as pd

    return pd.DataFrame([
        # ordinary region, fine
        dict(solver="newton", guess="simple", e=0.5, M=1.0,
             converged=True, failure=None, error=1e-15),
        # hard corner, ran out of iterations ("slow")
        dict(solver="newton", guess="simple", e=0.99, M=0.01,
             converged=False, failure=None, error=1e-2),
        # hard corner, blew up ("broken")
        dict(solver="newton", guess="simple", e=0.99, M=0.05,
             converged=False, failure="OverflowError", error=None),
        # hard corner, converged to the wrong root
        dict(solver="newton", guess="simple", e=0.95, M=0.02,
             converged=True, failure=None, error=1e-3),
    ])


def test_failure_rates_separates_slow_from_broken():
    from keplerbench.evaluation.robustness import failure_rates

    with skip_if_unimplemented():
        row = failure_rates(_fake_results()).iloc[0]
        assert row["n_points"] == 4
        assert row["n_converged"] == 2
        assert row["n_max_iter_hit"] == 1
        assert row["n_exception"] == 1
        assert row["failure_rate"] == pytest.approx(0.5)


def test_failure_rates_by_region_finds_the_hard_corner():
    from keplerbench.evaluation.robustness import failure_rates_by_region

    with skip_if_unimplemented():
        out = failure_rates_by_region(_fake_results()).set_index("region")
        assert out.loc["hard_corner", "n_points"] == 3
        assert out.loc["ordinary", "failure_rate"] == pytest.approx(0.0)


def test_wrong_root_rate_uses_the_error_not_the_residual():
    from keplerbench.evaluation.robustness import wrong_root_rate

    with skip_if_unimplemented():
        row = wrong_root_rate(_fake_results()).iloc[0]
        assert row["n_converged"] == 2
        assert row["n_wrong_root"] == 1
        assert row["wrong_root_rate"] == pytest.approx(0.5)
