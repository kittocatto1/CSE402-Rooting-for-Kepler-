"""Tests for Newton, Danby and Markley. Owner: Dipit."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.runner import solve_one


def test_newton_converges_quadratically_on_an_easy_case():
    """Residual should roughly square each iteration once in the asymptotic
    regime - a cheap smoke test that the update rule is right."""
    r = solve_one(get_solver("newton"), get_guess("simple"),
                  e=0.3, M=1.0, tol=0.0, max_iter=6, record_history=True)
    res = [rec.residual for rec in r.history if rec.residual > 1e-15]
    assert len(res) >= 3
    # log of residual should roughly double in magnitude each step
    assert res[2] < res[1] ** 1.5


def test_newton_costs_one_sincos_pair_per_iteration():
    """The Section 4.1 claim for the baseline, checked mechanically."""
    r = solve_one(get_solver("newton"), get_guess("simple"),
                  e=0.5, M=1.0, tol=0.0, max_iter=4, record_history=True)
    # 4 iterations, 1 sincos pair each. The extra sin_only calls come from
    # the residual check the base class does, which every solver pays.
    assert r.cost["sincos_pairs"] == 4


@pytest.mark.parametrize("solver_name", ["danby", "markley"])
def test_matches_a_brute_force_root(solver_name, sample_points):
    """Compare against scipy's bracketing solver - independent of our code."""
    from scipy.optimize import brentq

    solver = get_solver(solver_name)
    guess = get_guess("simple")
    with skip_if_unimplemented():
        for e, M in sample_points:
            expected = brentq(lambda E: E - e * math.sin(E) - M,
                              M - 1.1, M + 1.1, xtol=1e-15, rtol=1e-15)
            r = solve_one(solver, guess, e=e, M=M, tol=1e-14, max_iter=100)
            assert r.E == pytest.approx(expected, abs=1e-10), (solver_name, e, M)


def test_danby_reaches_tolerance_in_few_iterations():
    """Order 4 should not need many steps from a decent start.
    TODO(Dipit): tighten this bound once the implementation is in and you
    know the real iteration counts - do not loosen it to make it pass."""
    with skip_if_unimplemented():
        r = solve_one(get_solver("danby"), get_guess("simple"),
                      e=0.5, M=1.0, tol=1e-14, max_iter=50)
        assert r.converged and r.iterations <= 6


def test_markley_is_not_iterative():
    with skip_if_unimplemented():
        r = solve_one(get_solver("markley"), get_guess("simple"),
                      e=0.5, M=1.0, tol=1e-14)
        assert r.iterations == 0
