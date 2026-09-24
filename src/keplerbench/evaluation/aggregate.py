"""Turn raw per-solve rows into the summary tables the report needs.

Input everywhere is ``results/<experiment>/raw.csv`` as written by
``io.results_io.save_results``: one row per (solver, guess, grid point).

Two conventions run through this module.

*Medians are the headline, and the tail is kept beside them.*  The
interesting behaviour lives in the pathological corner, which is a small
fraction of any grid; a mean over the whole sweep averages it away, so every
median here is reported next to the worst case it came from.

*A blank cell means "not measured".*  Where a number depends on work another
track has not finished - Dipit's cost weights, Suchi's measured orders - the
column is produced as NaN rather than dropped or filled in by hand, so a
half-finished table still lines up with the finished one.

Owner: Anisa.
"""

from __future__ import annotations

import warnings

import pandas as pd

from keplerbench.core.registry import get_solver, list_solvers
from keplerbench.evaluation.metrics import correct_digits_from_error
from keplerbench.evaluation.robustness import region
from keplerbench.experiments.runner import GUESS_INDEPENDENT

#: Guess label the runner uses for closed-form solvers, which ignore E0.
GUESS_INDEPENDENT_LABEL = "n/a"

#: Baseline the other guesses are measured against in guess_layer_effect.
BASELINE_GUESS = "simple"


# ----------------------------------------------------------------------
# Shared preparation
# ----------------------------------------------------------------------
def _weighted_costs(df: pd.DataFrame) -> pd.Series:
    """Cost of each solve in "one plain sin() call" units.

    The counters are in the ``cost_*`` columns; turning them into one number
    needs the weights Dipit measures. While ``cost_model`` is still a stub -
    or while its default weights are still NaN - this returns an all-NaN
    column so the summary keeps its shape and the empty cells say plainly
    that the measurement has not been made.
    """
    blank = pd.Series(float("nan"), index=df.index, name="weighted_cost")
    counter_columns = [c for c in df.columns if c.startswith("cost_")]
    if not counter_columns:
        return blank

    try:
        from keplerbench.evaluation.cost_model import DEFAULT_WEIGHTS, weighted_cost
        counts = df[counter_columns].rename(columns=lambda c: c[len("cost_"):])
        costs = counts.apply(
            lambda row: weighted_cost(row.to_dict(), DEFAULT_WEIGHTS), axis=1)
    except NotImplementedError:
        warnings.warn(
            "evaluation.cost_model.weighted_cost is still a stub; the cost "
            "columns of this summary will be blank",
            stacklevel=3,
        )
        return blank
    return costs.rename("weighted_cost")


def _restore_guess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Undo pandas reading the "n/a" pseudo-guess back as NaN.

    ``io.results_io.load_results`` already prevents this, but a DataFrame
    built by a plain ``pd.read_csv`` still arrives with the closed-form
    solvers' guess as NaN - and ``groupby`` drops NaN keys, so those solvers
    would disappear from the summary without a word. Same rule as
    ``plotting.grid_plots._restore_guess_labels``, so a table and a figure
    can never disagree about which rows exist: only guess-independent
    solvers may lack a label, and anything else is a real defect.
    """
    missing = df["guess"].isna()
    if not missing.any():
        return df
    unexplained = sorted(set(
        df.loc[missing & ~df["solver"].isin(GUESS_INDEPENDENT), "solver"].astype(str)))
    if unexplained:
        raise ValueError(
            f"rows with no guess label for {unexplained}; only the "
            f"guess-independent solvers {sorted(GUESS_INDEPENDENT)} may lack one"
        )
    df = df.copy()
    df.loc[missing, "guess"] = GUESS_INDEPENDENT_LABEL
    return df


#: What _prepare guarantees downstream, so an empty sweep still has a shape.
_PREPARED_COLUMNS = ("solver", "guess", "e", "M", "converged", "iterations",
                     "residual", "error", "wall_time",
                     "correct_digits", "weighted_cost")

#: The statistics every summary table carries, beside its grouping keys.
SUMMARY_COLUMNS = ("n_points", "n_converged", "converged_frac",
                   "mean_iterations", "median_iterations", "max_iterations",
                   "mean_weighted_cost", "median_weighted_cost",
                   "median_wall_time",
                   "median_correct_digits", "worst_correct_digits")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived per-solve columns every summary below shares.

    An empty sweep is a real state, not an error - every solver may still be
    a skeleton, or the grid may have been truncated to nothing - so it yields
    an empty table with the right columns rather than a KeyError.
    """
    if df.empty or "solver" not in df.columns:
        return pd.DataFrame(columns=list(_PREPARED_COLUMNS))

    out = _restore_guess_labels(df).copy()
    out["converged"] = out["converged"].astype(bool)
    out["correct_digits"] = out["error"].map(correct_digits_from_error)
    out["weighted_cost"] = _weighted_costs(out)
    if "wall_time" not in out.columns:
        out["wall_time"] = float("nan")
    return out


def _summarise(prepared: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One summary row per group.

    Iteration and cost statistics are taken over the CONVERGED rows only. A
    run that gave up at ``max_iter`` did not "take 50 iterations" in any
    sense worth averaging - mixing those in makes a solver look slow when it
    actually failed, which is a different finding. ``converged_frac`` on the
    same row says how much of the group was set aside.
    """
    if prepared.empty:
        return pd.DataFrame(columns=list(keys) + list(SUMMARY_COLUMNS))

    rows = []
    for key, group in prepared.groupby(keys, sort=True):
        done = group[group["converged"]]
        digits = group["correct_digits"].dropna()
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row.update({
            "n_points": len(group),
            "n_converged": len(done),
            "converged_frac": len(done) / len(group) if len(group) else float("nan"),
            "mean_iterations": done["iterations"].mean(),
            "median_iterations": done["iterations"].median(),
            "max_iterations": done["iterations"].max(),
            "mean_weighted_cost": done["weighted_cost"].mean(),
            "median_weighted_cost": done["weighted_cost"].median(),
            "median_wall_time": done["wall_time"].median(),
            "median_correct_digits": digits.median(),
            # The worst point, not the worst converged point: a solver that
            # lands far from the root is exactly what this column is for.
            "worst_correct_digits": digits.min(),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.astype({"n_points": int, "n_converged": int}) if len(out) else out


# ----------------------------------------------------------------------
# The report's tables
# ----------------------------------------------------------------------
def summarise_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (solver, guess) over the whole grid."""
    return _summarise(_prepare(df), ["solver", "guess"])


def summarise_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Same summary, split into ordinary region vs pathological corner.

    The region labels come from ``robustness.region`` rather than being
    redefined here, so this table and the failure-rate table always mean the
    same thing by "hard_corner".
    """
    prepared = _prepare(df)
    prepared["region"] = region(prepared)
    return _summarise(prepared, ["solver", "guess", "region"])


def guess_layer_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Isolate the effect of the starting guess, averaged over solvers.

    The proposal insists "better start" and "better iteration" must never be
    conflated. This table is how we show they were not: for each solver, the
    change in mean iterations and mean cost relative to the ``simple`` guess.

    Two things keep the comparison honest:

    * Only grid points that converged under BOTH the baseline guess and the
      compared guess enter a row. Averaging over whatever happened to
      converge would let a guess look good precisely because it failed on the
      hard points and so never paid for them.
    * Closed-form solvers are dropped. They ignore E0 and are run once under
      the ``"n/a"`` label, so they have no guess effect to report.

    The last two columns are the interaction the proposal asks about: the
    spread of one guess's delta ACROSS solvers. Near zero means the guess
    helps everything equally; a wide spread means the benefit depends on
    which iteration it feeds.
    """
    prepared = _prepare(df)
    prepared = prepared[~prepared["solver"].isin(GUESS_INDEPENDENT)]
    prepared = prepared[prepared["guess"] != GUESS_INDEPENDENT_LABEL]

    rows = []
    for solver, per_solver in prepared.groupby("solver", sort=True):
        baseline = per_solver[per_solver["guess"] == BASELINE_GUESS]
        if baseline.empty:
            warnings.warn(
                f"guess_layer_effect: no {BASELINE_GUESS!r} rows for solver "
                f"{solver!r}; it has no baseline to compare against and is "
                "left out of the table",
                stacklevel=2,
            )
            continue
        baseline = baseline.set_index(["e", "M"])

        for guess, per_guess in per_solver.groupby("guess", sort=True):
            per_guess = per_guess.set_index(["e", "M"])
            shared = baseline.index.intersection(per_guess.index)
            both = (baseline.loc[shared, "converged"]
                    & per_guess.loc[shared, "converged"])
            points = shared[both]
            if len(points) == 0:
                continue

            base_iter = baseline.loc[points, "iterations"].mean()
            base_cost = baseline.loc[points, "weighted_cost"].mean()
            mean_iter = per_guess.loc[points, "iterations"].mean()
            mean_cost = per_guess.loc[points, "weighted_cost"].mean()
            rows.append({
                "solver": solver,
                "guess": guess,
                "n_common_points": len(points),
                "baseline_mean_iterations": base_iter,
                "mean_iterations": mean_iter,
                "delta_iterations": mean_iter - base_iter,
                "baseline_mean_weighted_cost": base_cost,
                "mean_weighted_cost": mean_cost,
                "delta_weighted_cost": mean_cost - base_cost,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Interaction: does this guess help every solver by the same amount?
    across = out.groupby("guess")["delta_iterations"]
    out["delta_iterations_across_solvers_mean"] = out["guess"].map(across.mean())
    out["delta_iterations_across_solvers_spread"] = out["guess"].map(across.std())
    return out.sort_values(["solver", "guess"], ignore_index=True)


def methods_table(order_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reproduce the proposal's method table with MEASURED numbers added.

    Columns: category | method | claimed order | measured order |
             eval points/iter | sincos pairs/iter | reference

    The claimed order, category and reference are read off the solver classes
    through the registry, so the table cannot drift from the code. The
    measured order comes from ``order_df`` (Suchi's verification summary) and
    the per-iteration counts from ``cost_model.per_iteration_cost_table()``
    (Dipit's, measured from the instrumented run).

    Any cell whose underlying run has not been done is left blank. Nothing
    here is ever typed in from a paper.
    """
    measured: dict[str, float] = {}
    if order_df is not None and not order_df.empty:
        if {"solver", "measured_order"} <= set(order_df.columns):
            measured = (order_df.groupby("solver")["measured_order"]
                                .median().to_dict())
        else:
            warnings.warn(
                "methods_table: order_df has no solver/measured_order columns; "
                "the measured-order column will be blank",
                stacklevel=2,
            )

    costs: dict[str, dict] = {}
    try:
        from keplerbench.evaluation.cost_model import per_iteration_cost_table
        costs = {str(row["method"]): row for row in per_iteration_cost_table()}
    except NotImplementedError:
        warnings.warn(
            "evaluation.cost_model.per_iteration_cost_table is still a stub; "
            "the per-iteration cost columns will be blank",
            stacklevel=2,
        )

    rows = []
    for name in list_solvers():
        solver = get_solver(name)
        cost = costs.get(name, {})
        rows.append({
            "category": solver.category,
            "method": name,
            "claimed_order": solver.theoretical_order,
            "measured_order": measured.get(name, float("nan")),
            "eval_points_per_iter": cost.get("eval points", float("nan")),
            "sincos_pairs_per_iter": cost.get("sincos pairs", float("nan")),
            "reference": solver.reference,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["category", "method"], ignore_index=True)
