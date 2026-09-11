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
