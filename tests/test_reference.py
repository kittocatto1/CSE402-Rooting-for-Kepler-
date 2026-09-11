"""Tests for the high-precision ground truth. Owner: Anisa."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.reference.mpmath_reference import reference_root


def test_reference_root_satisfies_the_equation(sample_points):
    with skip_if_unimplemented():
        for e, M in sample_points:
            E = reference_root(e, M)
            assert abs(E - e * math.sin(E) - M) < 1e-14, (e, M)


def test_reference_root_is_exact_at_zero_eccentricity():
    with skip_if_unimplemented():
        assert reference_root(0.0, 1.234) == pytest.approx(1.234, abs=1e-15)


def test_reference_root_does_not_use_a_solver_under_test():
    """Guard against someone 'simplifying' this to call Newton - the ground
    truth must stay independent of the methods being benchmarked."""
    import inspect

    from keplerbench.reference import mpmath_reference

    src = inspect.getsource(mpmath_reference)
    for banned in ("get_solver", "NewtonSolver", "DanbySolver", "solve_one"):
        assert banned not in src, f"reference must not depend on {banned}"
