"""Tests for the metric layer. Owners: Suchi (order), Dipit (cost), Anisa."""

from __future__ import annotations

import math

import mpmath as mp
import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.types import IterationRecord
from keplerbench.evaluation.convergence_order import (
    _working_floor,
    acoc,
    coc,
    last_finite,
    order_from_history_detail,
)
from keplerbench.evaluation.cost_model import (
    COST_TABLE_COLUMNS,
    DEFAULT_WEIGHTS,
    cost_per_correct_digit,
    measure_weights_with_spread,
    per_iteration_cost_table,
    weighted_cost,
)
from keplerbench.evaluation.metrics import efficiency_index


def test_coc_recovers_a_known_order():
    """Build a synthetic error sequence with e_{n+1} = C * e_n**p and check
    the estimator returns p. If it cannot do this, it cannot be trusted on
    real data.

    Note on the last term: at p = 3 this sequence reaches 9.1e-175 by the
    fifth step, and the sixth (7.5e-523) underflows to exactly 0.0 in double
    precision. coc() therefore reports nan for that final triple - correctly,
    and by design: undefined estimates are returned in place rather than
    dropped, so the caller can see where the sequence stopped being
    informative. Hence last_finite() rather than [-1]. The underflow is a
    miniature of the precision floor that motivates the whole module.
    """
    with skip_if_unimplemented():
        p_true, C = 3.0, 0.5
        errs = [1e-2]
        for _ in range(5):
            errs.append(C * errs[-1] ** p_true)
        estimates = coc(errs)
        assert math.isnan(estimates[-1]), "underflowed term must report nan"
        assert last_finite(estimates) == pytest.approx(p_true, rel=1e-3)


def test_coc_at_extended_precision_has_no_floor():
    """The same sequence in mpmath: no underflow, so every triple is usable.

    This is the path experiments/verification.py runs on, and the reason the
    estimators are written to be arithmetic-agnostic.
    """
    with skip_if_unimplemented():
        with mp.workdps(200):
            p_true = mp.mpf(3)
            errs = [mp.mpf("1e-2")]
            for _ in range(5):
                errs.append(mp.mpf("0.5") * errs[-1] ** p_true)
            estimates = coc(errs)
            assert len(estimates) == 4
            assert all(not mp.isnan(v) for v in estimates)
            assert float(estimates[-1]) == pytest.approx(3.0, rel=1e-12)


def test_precision_floor_stays_in_the_active_arithmetic():
    """A zero-valued or absent scale must not drop the floor to a double.

    The floor is relative to |root|. When the root sits at zero - or the
    history is empty - there is no scale to take it from, and the fallback
    used to be a Python ``1``, which silently produced a double-precision
    floor of 2.2e-14 even inside a 2000-digit context. Every mpf error below
    that was discarded and the run came back "unmeasurable", roughly 1980
    orders of magnitude too early.
    """
    with skip_if_unimplemented():
        with mp.workdps(2000):
            tiny = mp.mpf(10) ** -50
            for scale in (0, None):
                floor = _working_floor(scale, like=tiny)
                assert isinstance(floor, mp.mpf), f"{scale!r} left mp arithmetic"
                assert floor < mp.mpf(10) ** -1900, mp.nstr(floor, 6)
        # The float path must be untouched by the fix.
        assert _working_floor(0, like=1e-20) == pytest.approx(2.220446049250313e-14)


def test_order_survives_an_iterate_that_blows_up_after_converging():
    """A divergent tail must not set an impossibly high floor.

    Taking the scale from the LAST iterate means one blow-up to 1e64 puts the
    floor at 2.2e50 - above every genuine error - so a clean order-2 run is
    reported as unmeasurable. Solvers really do this in the pathological
    corner, so the scale comes from the closest approach to the root instead.
    """
    with skip_if_unimplemented():
        errors = [1e-2, 1e-4, 1e-8, 1e-16, 1e-16]
        iterates = [1.3, 1.36, 1.365, 1.3652, 1e64]      # last one diverged
        history = [
            IterationRecord(iteration=i, E=iterates[i],
                            residual=errors[i], error=errors[i])
            for i in range(len(errors))
        ]
        estimate = order_from_history_detail(history)
        assert estimate.ok(), estimate.reason
        assert estimate.value == pytest.approx(2.0, abs=1e-9)
        assert estimate.n_usable == 3


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


def test_weighted_cost_defaults_to_measured_weights():
    counts = {"sincos_pairs": 1, "sin_only": 2}
    expected = DEFAULT_WEIGHTS["sincos_pairs"] + 2 * DEFAULT_WEIGHTS["sin_only"]
    assert weighted_cost(counts) == pytest.approx(expected)


def test_default_weights_are_measured_not_blank():
    """aggregate.py blanks the whole cost column while any weight is NaN."""
    assert DEFAULT_WEIGHTS["sin_only"] == 1.0
    assert all(math.isfinite(w) and w > 0 for w in DEFAULT_WEIGHTS.values())


def test_weighted_cost_zero_count_never_pays_an_unmeasured_weight():
    weights = {"sincos_pairs": 1.0, "sin_only": 1.0,
               "synthesised_derivatives": float("nan")}
    assert weighted_cost({"sincos_pairs": 1, "synthesised_derivatives": 0},
                         weights) == pytest.approx(1.0)


def test_cost_per_correct_digit():
    weights = {"sincos_pairs": 2.0, "sin_only": 1.0}
    counts = {"sincos_pairs": 3, "sin_only": 4}          # cost 10
    assert cost_per_correct_digit(counts, 1e-10, weights) == pytest.approx(1.0)
    # capped at the double-precision ceiling, like correct_digits
    assert cost_per_correct_digit(counts, 1e-20, weights) == pytest.approx(10 / 16)


def test_cost_per_correct_digit_counts_an_exact_hit_at_the_ceiling():
    """Most converged solves land exactly on the reference; dropping them
    would delete each solver's best results from the ratio."""
    weights = {"sincos_pairs": 2.0, "sin_only": 1.0}
    counts = {"sincos_pairs": 3, "sin_only": 4}          # cost 10
    assert cost_per_correct_digit(counts, 0.0, weights) == pytest.approx(10 / 16)


@pytest.mark.parametrize("error", [-1e-3, float("nan"), float("inf"),
                                   1.0, 5.0, None])
def test_cost_per_correct_digit_is_nan_when_meaningless(error):
    assert math.isnan(cost_per_correct_digit({"sin_only": 1}, error))


def test_measure_weights_normalises_by_sin():
    weights, spread = measure_weights_with_spread(n_calls=80_000, repeats=3)
    assert weights["sin_only"] == 1.0
    assert set(weights) == set(DEFAULT_WEIGHTS) == set(spread)
    for key, (lo, hi) in spread.items():
        assert 0.0 < lo <= hi
        assert math.isfinite(weights[key]) and weights[key] > 0.0


def test_per_iteration_cost_table_is_measured_per_solver():
    from keplerbench.core.registry import list_solvers

    rows = {r["method"]: r for r in per_iteration_cost_table()}
    assert set(rows) == set(list_solvers())
    for row in rows.values():
        assert tuple(row) == COST_TABLE_COLUMNS
    # Newton and Danby touch one new point per iteration with one sincos
    # pair - Danby's f'' and f''' are free - which is the Section 4.1 point.
    for name in ("newton", "danby"):
        assert rows[name]["sincos pairs"] == 1
        assert rows[name]["eval points"] == 1
    # the with-memory methods recycle values into synthesised derivatives
    assert rows["nwm9"]["synthesised derivatives"] > 0
    assert rows["markley"]["theoretical order"] is None
