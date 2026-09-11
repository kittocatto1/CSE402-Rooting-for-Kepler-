"""Tests for the shared contract. Owner: Anisa."""

from __future__ import annotations

import math

import pytest

from keplerbench.core.counters import CostCounter
from keplerbench.core.types import KeplerProblem


def test_f_is_kepler_equation():
    p = KeplerProblem(e=0.5, M=0.3)
    E = 0.7
    assert p.f(E) == pytest.approx(E - 0.5 * math.sin(E) - 0.3)


def test_f_fprime_costs_one_sincos_pair():
    p = KeplerProblem(e=0.5, M=0.3)
    p.f_fprime(0.7)
    assert p.cost.sincos_pairs == 1
    assert p.cost.eval_points == 1
    assert p.cost.sin_only == 0


def test_repeated_point_not_double_counted():
    """Touching the same E twice is one evaluation POINT, not two."""
    p = KeplerProblem(e=0.5, M=0.3)
    p.f(0.7)
    p.fprime(0.7)
    assert p.cost.eval_points == 1
    assert p.cost.sin_only == 1
    assert p.cost.cos_only == 1


def test_derivatives_are_consistent():
    p = KeplerProblem(e=0.4, M=0.2)
    E = 1.1
    f, f1, f2, f3 = p.derivatives(E, order=3)
    assert f == pytest.approx(E - 0.4 * math.sin(E) - 0.2)
    assert f1 == pytest.approx(1 - 0.4 * math.cos(E))
    assert f2 == pytest.approx(0.4 * math.sin(E))
    assert f3 == pytest.approx(0.4 * math.cos(E))


def test_reset_clears_counters():
    c = CostCounter()
    c.note_point(1.0)
    c.sincos_pairs += 3
    c.reset()
    assert c.snapshot() == {
        "eval_points": 0, "sincos_pairs": 0, "sin_only": 0,
        "cos_only": 0, "synthesised_derivatives": 0, "extra_flops": 0,
    }
    # A previously visited point must count again after a reset.
    assert c.note_point(1.0) is True
