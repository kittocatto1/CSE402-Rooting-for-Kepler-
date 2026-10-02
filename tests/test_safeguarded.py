"""Tests for the safeguarded NWM solvers (solvers/safeguarded.py).

Three promises are checked:

  1. Away from the hard corner the guard never fires, so the guarded solver
     takes exactly the same steps as the published one.
  2. In the hard corner the guarded solver fails less often than the
     published one.
  3. The guard does not cost convergence order: at 2000 digits the guarded
     solvers still pass their papers' order check, and the memoryless
     controls fall back to exactly 8.
"""

from __future__ import annotations

import pytest

import keplerbench.guesses  # noqa: F401
import keplerbench.solvers  # noqa: F401
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.grid import pathological_grid
from keplerbench.experiments.runner import solve_one


@pytest.mark.parametrize("base", ["nwm9", "nwm11"])
def test_guard_is_silent_on_ordinary_points(base):
    guarded, published = get_solver(f"{base}_guarded"), get_solver(base)
    guess = get_guess("canonical")
    for e, M in [(0.1, 0.5), (0.5, 1.0), (0.7, 2.0), (0.3, 3.0)]:
        a = solve_one(guarded, guess, e, M, record_history=True)
        b = solve_one(published, guess, e, M, record_history=True)
        assert a.iterations == b.iterations
        assert a.E == b.E


@pytest.mark.parametrize("base", ["nwm9", "nwm11"])
def test_guard_reduces_corner_failures(base):
    guarded, published = get_solver(f"{base}_guarded"), get_solver(base)
    points = pathological_grid()
    failed = {"guarded": 0, "published": 0}
    for guess_name in ["simple", "canonical", "radvel", "napier"]:
        guess = get_guess(guess_name)
        for e, M in points:
            failed["guarded"] += not solve_one(guarded, guess, e, M).converged
            failed["published"] += not solve_one(published, guess, e, M).converged
    assert failed["guarded"] < failed["published"]


@pytest.mark.parametrize("name, functions_attr, expected", [
    ("nwm9_guarded", "NWM9_TEST_FUNCTIONS", 8.8989),
    ("nwm11_guarded", "PAPER_TEST_FUNCTIONS", 10.7446),
    ("nwm9_memoryless", "NWM9_TEST_FUNCTIONS", 8.0),
    ("nwm11_memoryless", "PAPER_TEST_FUNCTIONS", 8.0),
])
def test_order_is_kept(name, functions_attr, expected):
    from keplerbench.experiments import verification

    rows = verification.verify_order_on_test_functions(
        name, n_iterations=8, functions=getattr(verification, functions_attr))
    assert all(r["passed"] for r in rows)
    mean = sum(r["measured_order"] for r in rows) / len(rows)
    assert mean == pytest.approx(expected, rel=0.02)
