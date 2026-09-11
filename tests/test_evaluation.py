"""Tests for the metric layer. Owners: Suchi (order), Dipit (cost), Anisa."""

from __future__ import annotations

import pytest

from conftest import skip_if_unimplemented
from keplerbench.evaluation.convergence_order import acoc, coc
from keplerbench.evaluation.cost_model import weighted_cost
from keplerbench.evaluation.metrics import efficiency_index


def test_coc_recovers_a_known_order():
    """Build a synthetic error sequence with e_{n+1} = C * e_n**p and check
    the estimator returns p. If it cannot do this, it cannot be trusted on
    real data."""
    with skip_if_unimplemented():
        p_true, C = 3.0, 0.5
        errs = [1e-2]
        for _ in range(5):
            errs.append(C * errs[-1] ** p_true)
        estimates = coc(errs)
        assert estimates[-1] == pytest.approx(p_true, rel=1e-3)


def test_acoc_agrees_with_coc_on_the_same_sequence():
    with skip_if_unimplemented():
        p_true, C = 2.0, 0.3
        errs = [1e-2]
        for _ in range(6):
            errs.append(C * errs[-1] ** p_true)
        # iterates approaching 0 from above with those errors
        iterates = [0.0 + e for e in errs]
        assert acoc(iterates)[-1] == pytest.approx(p_true, rel=1e-2)


def test_efficiency_index():
    # Order 8 with 4 evaluations -> 8**(1/4)
    assert efficiency_index(8.0, 4.0) == pytest.approx(8.0 ** 0.25)


def test_weighted_cost_ignores_diagnostic_counters():
    """eval_points and extra_flops are diagnostics, not costs."""
    with skip_if_unimplemented():
        counts = {"sincos_pairs": 2, "sin_only": 1, "cos_only": 0,
                  "synthesised_derivatives": 0, "eval_points": 3,
                  "extra_flops": 999}
        weights = {"sincos_pairs": 1.5, "sin_only": 1.0, "cos_only": 1.0,
                   "synthesised_derivatives": 0.1}
        assert weighted_cost(counts, weights) == pytest.approx(4.0)
