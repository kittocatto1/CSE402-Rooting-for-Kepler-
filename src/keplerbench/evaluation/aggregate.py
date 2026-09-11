"""Turn raw per-solve rows into the summary tables the report needs.

Owner: Anisa.
"""

from __future__ import annotations

import pandas as pd


def summarise_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (solver, guess) over the whole grid.

    Columns to produce:
        solver, guess, n_points, converged_frac,
        mean_iterations, median_iterations, max_iterations,
        mean_weighted_cost, median_weighted_cost,
        median_wall_time,
        median_correct_digits, worst_correct_digits

    TODO(Anisa): groupby + agg. Use medians as the headline statistic and
    keep the max/worst columns - the tail behaviour in the hard corner is
    the interesting part and a mean hides it.
    """
    raise NotImplementedError("summarise_grid: see TODO above")


def summarise_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Same summary, split into ordinary region vs pathological corner.

    TODO(Anisa): reuse the region definition from
    evaluation.robustness.failure_rates_by_region so the two tables agree.
    """
    raise NotImplementedError("summarise_by_region: see TODO above")


def guess_layer_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Isolate the effect of the starting guess, averaged over solvers.

    The proposal insists "better start" and "better iteration" must never be
    conflated. This table is how we show they were not: for each guess,
    report the change in mean iterations and mean cost relative to the
    ``simple`` guess, for every solver.

    TODO(Anisa): pivot on guess, take ``simple`` as the baseline column, and
    report deltas. Also report the interaction - whether a given guess helps
    all solvers equally or only some.
    """
    raise NotImplementedError("guess_layer_effect: see TODO above")


def methods_table() -> pd.DataFrame:
    """Reproduce the proposal's method table with MEASURED numbers added.

    Columns: category | method | claimed order | measured order |
             eval points/iter | sincos pairs/iter | reference

    TODO(Anisa): pull claimed order and category off the solver classes via
    the registry, measured order from the verification results, and the cost
    columns from cost_model.per_iteration_cost_table(). Leave any cell blank
    if the underlying run has not been done - never fill it in by hand.
    """
    raise NotImplementedError("methods_table: see TODO above")
