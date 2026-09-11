"""Tests for NWM9 and NWM11. Owner: Suchi."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.runner import solve_one
from keplerbench.solvers._withmemory_base import (
    hermite_derivative_estimate,
    newton_divided_differences,
)


def test_divided_differences_of_a_polynomial_terminate():
    """For a cubic, the 4th divided difference must be zero."""
    with skip_if_unimplemented():
        f = lambda x: 2 * x**3 - x + 5
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        coeffs = newton_divided_differences(xs, [f(x) for x in xs])
        assert coeffs[3] == pytest.approx(2.0)      # leading coefficient
        assert coeffs[4] == pytest.approx(0.0, abs=1e-9)


def test_hermite_derivative_estimate_is_accurate():
    with skip_if_unimplemented():
        f = lambda x: math.exp(x)
        xs = [0.9, 0.95, 1.0, 1.05, 1.1]
        est = hermite_derivative_estimate(xs, [f(x) for x in xs], order=1, at=1.0)
        assert est == pytest.approx(math.e, rel=1e-4)


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_carries_memory_between_iterations(solver_name):
    """A with-memory method must actually accumulate history.

    TODO(Suchi): this checks the mechanism, not the maths. Once implemented,
    also assert that the accelerating parameters CHANGE between iteration 1
    and 2 - if they stay at their initial value, the memory is not wired up
    and the method silently degrades to its memoryless base.
    """
    solver = get_solver(solver_name)
    problem_state = None
    with skip_if_unimplemented():
        from keplerbench.core.types import KeplerProblem

        p = KeplerProblem(e=0.3, M=1.0)
        state = solver.init_state(p, 1.0)
        E = solver.step(p, 1.0, state)
        assert math.isfinite(E)
        assert state["memory"].has_memory(), "state was not updated by step()"
        problem_state = state
    assert problem_state is None or "memory" in problem_state


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_reaches_tolerance_on_kepler(solver_name, sample_points):
    solver = get_solver(solver_name)
    guess = get_guess("simple")
    with skip_if_unimplemented():
        for e, M in sample_points:
            r = solve_one(solver, guess, e=e, M=M, tol=1e-13, max_iter=100)
            assert r.converged, (solver_name, e, M, r.failure)


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_measured_order_matches_the_paper(solver_name):
    """Work Plan step 2, as a test. This is the gate before the grid run."""
    from keplerbench.experiments.verification import verify_order_on_test_functions

    with skip_if_unimplemented():
        out = verify_order_on_test_functions(solver_name)
        assert out, "no test functions configured - fill PAPER_TEST_FUNCTIONS"
        for row in out:
            assert row["passed"], row
