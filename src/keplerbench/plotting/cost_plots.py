"""Cost-accounting figures. Owner: Dipit.

These draw, they never measure.  The counters come from result files or
from ``evaluation.cost_model.per_iteration_cost_table``, and turning them
into one number always goes through ``cost_model.weighted_cost``, so a
figure and a table can never disagree about what a solve cost.

Expected columns
----------------
``cost_table_df``  one row per method, as ``per_iteration_cost_table()``:
                   method, eval points, sincos pairs, sin only, cos only,
                   synthesised derivatives, theoretical order
``df`` (raw.csv)   solver, guess, e, M, converged, error, cost_* counters
``timing_df``      solver, guess, e, M, seconds (timing.csv)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keplerbench.evaluation.cost_model import (CALL_COUNT_WEIGHTS,
                                               DEFAULT_WEIGHTS,
                                               cost_per_correct_digit,
                                               weighted_cost)
from keplerbench.evaluation.metrics import (correct_digits_from_error,
                                            efficiency_index)
from keplerbench.experiments.runner import GUESS_INDEPENDENT
from keplerbench.plotting.style import SOLVER_COLORS

__all__ = [
    "plot_cost_breakdown",
    "plot_cost_vs_accuracy",
    "plot_wallclock_vs_cost",
]

#: Evaluation kinds in stacking order, as (counter, table column, label).
_KINDS = (
    ("sincos_pairs", "sincos pairs", "sincos pair"),
    ("sin_only", "sin only", "sin only"),
    ("cos_only", "cos only", "cos only"),
    ("synthesised_derivatives", "synthesised derivatives", "synthesised"),
)
_KIND_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#8C8C8C")

#: Marker per guess in scatter plots (line styles do not show on points).
_GUESS_MARKERS = {"simple": "o", "canonical": "s", "radvel": "^",
                  "napier": "D", "n/a": "*"}

#: The closed-form solvers' pseudo-guess, see experiments.runner.
GUESS_INDEPENDENT_LABEL = "n/a"


def _require(df: pd.DataFrame, columns, who: str) -> None:
    """Fail loudly on a missing column - an empty figure is worse."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{who}: DataFrame is missing {missing}; got {list(df.columns)}")


def _restore_guess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Give the closed-form solvers back their "n/a" guess if it read as NaN."""
    if "guess" not in df.columns or not df["guess"].isna().any():
        return df
    df = df.copy()
    fill = df["guess"].isna() & df["solver"].isin(GUESS_INDEPENDENT)
    df.loc[fill, "guess"] = GUESS_INDEPENDENT_LABEL
    return df


def _row_costs(df: pd.DataFrame, weights) -> pd.Series:
    """weighted_cost of every row, from its cost_* counter columns."""
    counter_columns = [c for c in df.columns if c.startswith("cost_")]
    if not counter_columns:
        raise KeyError("no cost_* counter columns to cost")
    counts = df[counter_columns].rename(columns=lambda c: c[len("cost_"):])
    return counts.apply(lambda row: weighted_cost(row.to_dict(), weights), axis=1)


def _solver_order(names) -> list[str]:
    """Report order: the fixed palette order, then anything unknown."""
    known = [s for s in SOLVER_COLORS if s in set(names)]
    return known + sorted(set(names) - set(known))


# ----------------------------------------------------------------------
def plot_cost_breakdown(cost_table_df: pd.DataFrame) -> plt.Figure:
    """Stacked bars: per-iteration cost split by evaluation kind.

    One bar per solver, stacked into sincos pairs / sin only / cos only /
    synthesised. Two panels, because the report carries two costings: the
    weights measured on our CPython harness, and the one-call-per-point
    costing of a fused-sincos implementation (see ``cost_model``). The
    Section 4.1 argument is the difference between them.
    """
    _require(cost_table_df, ["method"] + [col for _, col, _ in _KINDS],
             "plot_cost_breakdown")
    table = cost_table_df.set_index("method").loc[
        _solver_order(cost_table_df["method"])]
    x = np.arange(len(table))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    panels = ((axes[0], DEFAULT_WEIGHTS, "Measured weights (CPython harness)"),
              (axes[1], CALL_COUNT_WEIGHTS, "Call count (fused sincos, C)"))
    for ax, weights, title in panels:
        bottom = np.zeros(len(table))
        for (counter, column, label), color in zip(_KINDS, _KIND_COLORS):
            height = table[column].astype(float).to_numpy() * weights[counter]
            ax.bar(x, height, bottom=bottom, color=color, label=label, width=0.6)
            bottom += height
        for xi, total in zip(x, bottom):
            ax.text(xi, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, table.index)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(bottom.max(), 1e-9) * 1.15)
    axes[0].set_ylabel("cost per iteration [sin() calls]")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.text(0.5, -0.02, "Markley: whole solve (not iterative). "
             "Counts include the shared residual check, one sin per iteration.",
             ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cost_vs_accuracy(df: pd.DataFrame,
                          cost_table_df: pd.DataFrame | None = None) -> plt.Figure:
    """Weighted cost against digits of accuracy achieved - the money plot.

    Left: one point per (solver, guess) at the median weighted cost and
    the mean correct digits over the grid, with the 10-90% cost range as a
    horizontal bar and a vertical bar down to the worst decile of digits.
    At a fixed tolerance nearly every solve hits the 16-digit ceiling, so
    the median digit count carries no information; the mean and the tail
    are where solvers differ. Further left at the same height is better.

    Right: the generic efficiency index against the measured median cost
    per correct digit. The index uses the claimed order and the calls per
    iteration with NO sincos discount (a pair counts as two, as in the
    source papers). If the two costings agreed, higher index would always
    mean lower cost; where a solver breaks that trend the generic index is
    wrong about Kepler's equation.

    ``cost_table_df`` defaults to ``per_iteration_cost_table()``.
    """
    _require(df, ["solver", "guess", "error"], "plot_cost_vs_accuracy")
    df = _restore_guess_labels(df).copy()
    if "converged" in df.columns:
        df = df[df["converged"].astype(bool)]
    df["weighted_cost"] = _row_costs(df, DEFAULT_WEIGHTS)
    df["digits"] = df["error"].map(correct_digits_from_error)
    df = df[np.isfinite(df["digits"])]
    if df.empty:
        raise ValueError("plot_cost_vs_accuracy: no converged rows with a "
                         "reference error - was the grid run with reference roots?")

    if cost_table_df is None:
        from keplerbench.evaluation.cost_model import per_iteration_cost_table
        cost_table_df = pd.DataFrame(per_iteration_cost_table())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for (solver, guess), group in df.groupby(["solver", "guess"], sort=False):
        cost, digits = group["weighted_cost"], group["digits"]
        med_x, mean_y = cost.median(), digits.mean()
        ax.errorbar(med_x, mean_y,
                    xerr=[[med_x - cost.quantile(0.1)], [cost.quantile(0.9) - med_x]],
                    yerr=[[max(mean_y - digits.quantile(0.1), 0.0)], [0.0]],
                    fmt=_GUESS_MARKERS.get(guess, "o"), ms=6, capsize=2, lw=0.8,
                    color=SOLVER_COLORS.get(solver, "k"), alpha=0.9)
    solver_handles = [plt.Line2D([], [], color=SOLVER_COLORS.get(s, "k"), marker="o",
                                 ls="", label=s)
                      for s in _solver_order(df["solver"])]
    guess_handles = [plt.Line2D([], [], color="0.4", marker=_GUESS_MARKERS.get(g, "o"),
                                ls="", label=g)
                     for g in sorted(df["guess"].unique(),
                                     key=list(_GUESS_MARKERS).index)]
    ax.legend(handles=solver_handles + guess_handles, fontsize=7, ncol=2)
    ax.set_xlabel("weighted cost per solve [sin() calls]")
    ax.set_ylabel("mean correct digits (bar: worst 10%)")
    ax.set_title("Cost vs accuracy (median cost, 10-90% range)", fontsize=10)

    # Right panel: generic index vs Kepler-specific cost per digit.
    per_digit = df.apply(lambda r: cost_per_correct_digit(
        {c[len("cost_"):]: r[c] for c in df.columns if c.startswith("cost_")},
        r["error"]), axis=1)
    df["cost_per_digit"] = per_digit
    exact_share = (df["error"] == 0.0).groupby(df["solver"]).mean()
    table = cost_table_df.set_index("method")
    skipped = []
    for solver in _solver_order(df["solver"]):
        order = table["theoretical order"].get(solver) if solver in table.index else None
        if order is None or not np.isfinite(float(order)):
            skipped.append(solver)
            continue
        row = table.loc[solver]
        calls = 2 * row["sincos pairs"] + row["sin only"] + row["cos only"]
        index = efficiency_index(float(order), float(calls))
        cpd = df.loc[df["solver"] == solver, "cost_per_digit"].median()
        ax2.scatter(index, cpd, s=50, color=SOLVER_COLORS.get(solver, "k"), zorder=3)
        ax2.annotate(f"{solver} ({100 * exact_share.get(solver, 0.0):.0f}% exact)",
                     (index, cpd), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax2.set_xlabel("efficiency index  order^(1/calls per iter)")
    ax2.set_ylabel("median cost per correct digit")
    title = "Generic index vs Kepler costing"
    if skipped:
        title += f"\n(no claimed order: {', '.join(skipped)})"
    ax2.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_wallclock_vs_cost(df: pd.DataFrame,
                           raw_df: pd.DataFrame | None = None) -> plt.Figure:
    """Measured wall-clock time against the weighted cost model.

    ``df`` is timing.csv. It carries no counters, so each timed point is
    matched to its row of ``raw_df`` (raw.csv) on solver, guess, e, M to get
    the modelled cost. If ``df`` already has cost_* columns, ``raw_df`` is
    not needed.

    If the cost model is any good these correlate; the Pearson r is in the
    title. If they do not, the weights are wrong or Python overhead
    dominates - either way it is reported, not hidden.
    """
    _require(df, ["solver", "guess", "e", "M", "seconds"], "plot_wallclock_vs_cost")
    timing = _restore_guess_labels(df)
    if not any(c.startswith("cost_") for c in timing.columns):
        if raw_df is None:
            raise KeyError("plot_wallclock_vs_cost: timing table has no cost_* "
                           "columns; pass raw_df (raw.csv) to supply them")
        raw = _restore_guess_labels(raw_df)
        counters = [c for c in raw.columns if c.startswith("cost_")]
        timing = timing.merge(raw[["solver", "guess", "e", "M"] + counters],
                              on=["solver", "guess", "e", "M"], how="inner")
    if timing.empty:
        raise ValueError("plot_wallclock_vs_cost: no timed point matched a raw.csv row")
    timing = timing.copy()
    timing["weighted_cost"] = _row_costs(timing, DEFAULT_WEIGHTS)
    timing["micros"] = timing["seconds"] * 1e6

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for solver in _solver_order(timing["solver"]):
        group = timing[timing["solver"] == solver]
        ax.scatter(group["weighted_cost"], group["micros"], s=14, alpha=0.7,
                   color=SOLVER_COLORS.get(solver, "k"), label=solver)
    x, y = timing["weighted_cost"].to_numpy(float), timing["micros"].to_numpy(float)
    if len(timing) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="0.3", lw=1, ls="--",
                label=f"fit: {slope:.2f} us per sin() {intercept:+.1f} us")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"Wall-clock vs cost model (Pearson r = {r:.2f})", fontsize=10)
    else:
        ax.set_title("Wall-clock vs cost model", fontsize=10)
    ax.set_xlabel("weighted cost per solve [sin() calls]")
    ax.set_ylabel("median wall-clock per solve [us]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
