"""Every solver must behave the same way from the outside.

These tests run against EVERY registered solver, so a new solver is checked
automatically.  They are the guarantee that the comparison is fair.

Owner: Anisa.
"""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, get_solver, list_guesses, list_solvers
from keplerbench.experiments.runner import solve_one

EXPECTED_SOLVERS = {"newton", "danby", "markley", "nwm9", "nwm11"}
EXPECTED_GUESSES = {"simple", "canonical", "radvel", "napier"}


def test_all_expected_methods_are_registered():
    assert EXPECTED_SOLVERS <= set(list_solvers())
    assert EXPECTED_GUESSES <= set(list_guesses())


@pytest.mark.parametrize("solver_name", sorted(EXPECTED_SOLVERS))
def test_solver_declares_its_metadata(solver_name):
    """The methods table in the report is generated from these attributes."""
    s = get_solver(solver_name)
    assert s.category != "uncategorised", "set .category"
    assert s.reference, "set .reference so the report can cite it"


@pytest.mark.parametrize("solver_name", sorted(EXPECTED_SOLVERS))
def test_solver_finds_the_root(solver_name, sample_points):
    """Whatever the method, the answer must satisfy Kepler's equation."""
    solver = get_solver(solver_name)
    guess = get_guess("simple")
    with skip_if_unimplemented():
        for e, M in sample_points:
            r = solve_one(solver, guess, e=e, M=M, tol=1e-13, max_iter=100)
            if not r.converged and solver_name != "markley":
                pytest.fail(f"{solver_name} did not converge at e={e}, M={M}")
            assert abs(r.E - e * math.sin(r.E) - M) < 1e-11, (e, M)


@pytest.mark.parametrize("solver_name", sorted(EXPECTED_SOLVERS))
def test_solver_records_cost(solver_name):
    """A solver that reports zero cost has bypassed the KeplerProblem API."""
    solver = get_solver(solver_name)
    guess = get_guess("simple")
    with skip_if_unimplemented():
        r = solve_one(solver, guess, e=0.5, M=1.0, tol=1e-13)
        assert r.cost["eval_points"] > 0
        total = (r.cost["sincos_pairs"] + r.cost["sin_only"]
                 + r.cost["cos_only"] + r.cost["synthesised_derivatives"])
        assert total > 0, "cost counters were never incremented"


@pytest.mark.parametrize("solver_name", sorted(EXPECTED_SOLVERS))
def test_solver_does_not_crash_in_the_hard_corner(solver_name):
    """Failure must be REPORTED, not raised - the robustness metric needs it."""
    solver = get_solver(solver_name)
    guess = get_guess("simple")
    with skip_if_unimplemented():
        r = solve_one(solver, guess, e=0.9999, M=1e-8, tol=1e-13, max_iter=200)
        assert r.failure is None or isinstance(r.failure, str)
