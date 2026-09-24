"""Tests for the per-solve metrics and the report's summary tables.

Owner: Anisa.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from keplerbench.core.types import IterationRecord, SolveResult
from keplerbench.evaluation.aggregate import (
    guess_layer_effect,
    methods_table,
    summarise_by_region,
    summarise_grid,
)
from keplerbench.evaluation.metrics import (
    DIGIT_CEILING,
    correct_digits,
    iterations_to_tolerance,
)


def _result(error=None, residuals=()):
    return SolveResult(
        solver="s", guess="g", e=0.1, M=0.2, E=0.0, converged=True,
        iterations=len(residuals), residual=0.0, error=error,
        history=[IterationRecord(iteration=i, E=0.0, residual=r)
                 for i, r in enumerate(residuals)],
    )


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def test_correct_digits_counts_digits():
    assert correct_digits(_result(error=1e-10)) == pytest.approx(10.0)


def test_correct_digits_is_capped_at_double_precision():
    """Without a cap a tiny error claims accuracy the arithmetic cannot hold."""
    assert correct_digits(_result(error=1e-30)) == DIGIT_CEILING


def test_an_exact_hit_scores_the_ceiling_not_nan():
    """error == 0 means the iterate matched the reference root to the last
    bit - the best measurable outcome. Reporting nan would drop a solver's
    best results out of every median and make it look worse than it is."""
    assert correct_digits(_result(error=0.0)) == DIGIT_CEILING


def test_correct_digits_is_nan_when_accuracy_is_unknown():
    assert math.isnan(correct_digits(_result(error=None)))      # no reference root
    assert math.isnan(correct_digits(_result(error=float("nan"))))
    assert math.isnan(correct_digits(_result(error=-1.0)))      # corrupt


def test_iterations_to_tolerance_finds_the_first_crossing():
    r = _result(residuals=[1e-1, 1e-5, 1e-11, 1e-16])
    assert iterations_to_tolerance(r, 1e-4) == 1
    assert iterations_to_tolerance(r, 1e-12) == 3


def test_iterations_to_tolerance_matches_the_solver_stopping_rule():
    """IterativeSolver._converged uses ``residual <= tol``; an exact hit on
    the tolerance must count here too, or metric and solver disagree."""
    assert iterations_to_tolerance(_result(residuals=[1e-1, 1e-5]), 1e-5) == 1


def test_iterations_to_tolerance_is_none_when_never_reached():
    assert iterations_to_tolerance(_result(residuals=[1e-1]), 1e-20) is None
    assert iterations_to_tolerance(_result(), 1e-12) is None     # no history


# ----------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------
def _grid(failures_in_corner=True):
    """A small sweep spanning both regions, with real non-convergence."""
    rows = []
    for solver, base in [("newton", 6), ("danby", 3)]:
        for guess, bump in [("simple", 2), ("canonical", 0)]:
            for e, M in [(0.3, 1.0), (0.5, 2.0), (0.95, 0.01), (0.99, 0.001)]:
                hard = e > 0.9 and M < 0.1
                failed = failures_in_corner and hard and solver == "newton" \
                    and guess == "simple"
                rows.append(dict(
                    solver=solver, guess=guess, e=e, M=M, E=1.0,
                    converged=not failed,
                    iterations=50 if failed else base + bump + (4 if hard else 0),
                    residual=1e-3 if failed else 1e-16,
                    error=1e-2 if failed else 1e-14,
                    wall_time=1e-5, failure=None,
                    cost_sincos_pairs=3, cost_eval_points=3))
    return pd.DataFrame(rows)


def test_summarise_grid_has_one_row_per_solver_guess():
    out = summarise_grid(_grid())
    assert len(out) == 4
    assert set(out["solver"]) == {"newton", "danby"}


def test_iteration_stats_exclude_runs_that_never_converged():
    """A run that gave up at max_iter did not "take 50 iterations" in any
    sense worth averaging - that is a failure, a different finding."""
    out = summarise_grid(_grid())
    row = out[(out["solver"] == "newton") & (out["guess"] == "simple")].iloc[0]
    assert row["n_points"] == 4 and row["n_converged"] == 2
    assert row["converged_frac"] == pytest.approx(0.5)
    assert row["mean_iterations"] == pytest.approx(8.0)     # not (8+8+50+50)/4


def test_the_failures_still_show_up_in_the_accuracy_columns():
    """Excluding them from the iteration mean must not hide them entirely."""
    out = summarise_grid(_grid())
    row = out[(out["solver"] == "newton") & (out["guess"] == "simple")].iloc[0]
    assert row["worst_correct_digits"] == pytest.approx(2.0)   # the 1e-2 error


def test_closed_form_solvers_are_not_dropped_by_the_n_a_label():
    """pandas reads "n/a" as NaN and groupby drops NaN keys, so markley can
    vanish from the summary without a word. See plotting.grid_plots, which
    guards the same hazard on the figure side."""
    df = _grid()
    df = pd.concat([df, pd.DataFrame([dict(
        solver="markley", guess=None, e=0.3, M=1.0, E=1.0, converged=True,
        iterations=0, residual=1e-16, error=1e-15, wall_time=1e-6,
        failure=None, cost_sincos_pairs=1, cost_eval_points=1)])],
        ignore_index=True)
    out = summarise_grid(df)
    assert "markley" in set(out["solver"])
    assert "n/a" in set(out["guess"])


def test_a_missing_guess_label_on_an_iterative_solver_fails_loudly():
    df = _grid()
    df.loc[df["solver"] == "danby", "guess"] = None
    with pytest.raises(ValueError, match="no guess label"):
        summarise_grid(df)


def test_summarise_by_region_separates_the_hard_corner():
    out = summarise_by_region(_grid())
    assert set(out["region"]) == {"ordinary", "hard_corner"}
    corner = out[(out["solver"] == "newton") & (out["guess"] == "simple")
                 & (out["region"] == "hard_corner")].iloc[0]
    ordinary = out[(out["solver"] == "newton") & (out["guess"] == "simple")
                   & (out["region"] == "ordinary")].iloc[0]
    # The whole point: the corner is where it fails, and an overall average
    # over the sweep would hide that.
    assert corner["converged_frac"] == 0.0
    assert ordinary["converged_frac"] == 1.0


def test_guess_layer_effect_compares_only_points_that_converged_under_both():
    """Otherwise a guess looks good precisely because it failed on the hard
    points and so never paid for them."""
    out = guess_layer_effect(_grid())
    newton = out[(out["solver"] == "newton") & (out["guess"] == "canonical")].iloc[0]
    assert newton["n_common_points"] == 2        # not 4 - the corner failed
    assert newton["delta_iterations"] == pytest.approx(-2.0)


def test_guess_layer_effect_reports_the_baseline_as_a_zero_delta():
    out = guess_layer_effect(_grid())
    for _, row in out[out["guess"] == "simple"].iterrows():
        assert row["delta_iterations"] == pytest.approx(0.0)


def test_guess_layer_effect_reports_the_across_solver_spread():
    """The proposal asks whether a guess helps all solvers equally."""
    out = guess_layer_effect(_grid())
    assert "delta_iterations_across_solvers_spread" in out.columns
    assert out["delta_iterations_across_solvers_mean"].notna().all()


def test_methods_table_reads_claims_off_the_solver_classes():
    out = methods_table()
    assert set(out["method"]) == {"newton", "danby", "markley", "nwm9", "nwm11"}
    newton = out[out["method"] == "newton"].iloc[0]
    assert newton["claimed_order"] == 2.0
    assert newton["category"] == "baseline"
    markley = out[out["method"] == "markley"].iloc[0]
    assert pd.isna(markley["claimed_order"])     # closed form - no asymptotic order


def test_methods_table_leaves_unmeasured_cells_blank():
    """Never filled in by hand from the papers."""
    out = methods_table()
    assert out["measured_order"].isna().all()    # verification has not run here


def test_methods_table_uses_measured_orders_when_they_exist():
    order_df = pd.DataFrame({"solver": ["newton", "newton", "danby"],
                             "measured_order": [1.98, 2.02, 3.9]})
    out = methods_table(order_df)
    assert out[out["method"] == "newton"].iloc[0]["measured_order"] == pytest.approx(2.0)
    assert out[out["method"] == "nwm9"].iloc[0]["measured_order"] != \
        out[out["method"] == "nwm9"].iloc[0]["measured_order"]        # still nan
