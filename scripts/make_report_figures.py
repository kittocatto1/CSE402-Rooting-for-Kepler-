#!/usr/bin/env python3
"""Regenerate every figure in the report from the result files.

Owner: Anisa (integration).

Run this LAST, after the experiments have written results/. It must not
compute any metric itself - it only loads result tables, hands them to
``evaluation/`` when a plotting function needs a derived table, and calls the
plotting functions. That way every figure in the report is reproducible with
one command and traceable to a result file.

Two rules about failure, because a stale figure in the report is worse than
a missing one:

  * Every figure is attempted, so one missing result file does not hide the
    state of all the others. The run ends with a report and a non-zero exit
    status if anything did not rebuild.
  * If a figure cannot be rebuilt, any previous copy of it is DELETED. An old
    PDF sitting in figures/ would otherwise be picked up by the report build
    and silently present last week's numbers as current.

Usage::

    python scripts/make_report_figures.py            # rebuild everything
    python scripts/make_report_figures.py --list     # just show the plan
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
import pandas as pd

matplotlib.use("Agg")           # no display on a lab machine or in CI
import matplotlib.pyplot as plt  # noqa: E402

from keplerbench.io.results_io import REPO_ROOT, load_results  # noqa: E402
from keplerbench.plotting import (  # noqa: E402
    convergence_plots,
    cost_plots,
    grid_plots,
    propagation_plots,
)
from keplerbench.plotting.style import FIGURE_DIR, save_figure, use_report_style  # noqa: E402

GRID = "grid_benchmark"
VERIFICATION = "verification"
PROPAGATION = "error_propagation"
MONTE_CARLO = "monte_carlo"


# ----------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------
@dataclass
class Figure:
    """One figure: what it is called, what it needs, and how to draw it.

    ``build`` returns a matplotlib Figure. It is only called once every path
    in ``needs`` exists, so a builder never has to check for its own inputs.
    """

    name: str
    section: str
    build: Callable[[], plt.Figure]
    needs: list[tuple[str, str]] = field(default_factory=list)

    def missing(self) -> list[str]:
        paths = [REPO_ROOT / "results" / exp / table for exp, table in self.needs]
        return [str(p.relative_to(REPO_ROOT)) for p in paths if not p.exists()]


def _endpoints(history: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """A typical (e, M) and the hardest one the history table actually holds.

    Chosen from the data rather than hard-coded, so the figure follows
    whatever sub-grid the config asked for instead of silently failing when
    someone changes it.
    """
    points = history[["e", "M"]].drop_duplicates()
    hard = points.sort_values(["e", "M"], ascending=[False, True]).iloc[0]
    ordinary = points[points["e"] <= points["e"].median()]
    typical = (ordinary if not ordinary.empty else points).iloc[len(ordinary) // 2]
    return (float(typical["e"]), float(typical["M"])), (float(hard["e"]), float(hard["M"]))


def _panels(n: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """A row-major grid of axes wide enough for ``n`` panels."""
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


#: Guess used for the solver-vs-solver figures. Showing every solver against
#: every guess at once puts twenty lines and a legend larger than the axes
#: into one panel, and the question that figure answers is about the solvers.
#: The guess layer has its own figure. ``"n/a"`` rides along so the
#: closed-form solver, which is run once under that label, is not dropped.
BASELINE_GUESSES = ("simple", "n/a")


def _residual_histories() -> plt.Figure:
    history = load_results(GRID, "history.csv")
    typical, hard = _endpoints(history)
    one_guess = history[history["guess"].isin(BASELINE_GUESSES)]
    return convergence_plots.plot_residual_history_pair(one_guess, typical, hard)


def _measured_vs_claimed_order() -> plt.Figure:
    order = load_results(VERIFICATION, "summary.csv")
    return convergence_plots.plot_measured_vs_claimed_order(order).figure


def _order_vs_eccentricity() -> plt.Figure:
    order = load_results(VERIFICATION, "summary.csv")
    return convergence_plots.plot_order_vs_eccentricity(order, log_1me=True).figure


def _iterations_heatmaps() -> plt.Figure:
    df = load_results(GRID, "raw.csv")
    pairs = sorted(df.groupby(["solver", "guess"]).groups)
    # One colour scale for every panel: per-panel scaling would let two
    # panels that look identical describe different iteration counts.
    vmax = grid_plots.shared_iteration_scale(df)
    fig, axes = _panels(len(pairs))
    for ax, (solver, guess) in zip(axes, pairs):
        grid_plots.plot_iterations_heatmap(df, solver, guess, ax=ax, vmax=vmax)
    fig.tight_layout()
    return fig


def _failure_maps() -> plt.Figure:
    df = load_results(GRID, "raw.csv")
    solvers = sorted(df["solver"].unique())
    fig, axes = _panels(len(solvers))
    for ax, solver in zip(axes, solvers):
        grid_plots.plot_failure_map(df, solver, ax=ax)
    fig.tight_layout()
    return fig


def _guess_effect() -> plt.Figure:
    return grid_plots.plot_guess_effect(load_results(GRID, "raw.csv")).figure


def _cost_breakdown() -> plt.Figure:
    from keplerbench.evaluation.cost_model import per_iteration_cost_table
    return cost_plots.plot_cost_breakdown(pd.DataFrame(per_iteration_cost_table()))


def _cost_vs_accuracy() -> plt.Figure:
    return cost_plots.plot_cost_vs_accuracy(load_results(GRID, "raw.csv"))


def _wallclock_vs_cost() -> plt.Figure:
    return cost_plots.plot_wallclock_vs_cost(load_results(GRID, "timing.csv"))


def _tolerance_vs_shift() -> plt.Figure:
    return propagation_plots.plot_tolerance_vs_parameter_shift(
        load_results(PROPAGATION, "summary.csv"))


def _error_budget() -> plt.Figure:
    """Solver-induced shift against measurement noise.

    The budget table is assembled by ``evaluation.propagation``, not here -
    this only reads the three result files it needs and passes them on.
    """
    from keplerbench.evaluation.propagation import compare_error_budgets

    shifts = load_results(PROPAGATION, "summary.csv")
    noise = pd.read_csv(REPO_ROOT / "results" / MONTE_CARLO / "summary.csv",
                        index_col=0).squeeze("columns").to_dict()
    sigma = pd.read_csv(REPO_ROOT / "results" / PROPAGATION / "posterior_sigma.csv",
                        index_col=0).squeeze("columns").to_dict()
    budget = compare_error_budgets(shifts, noise, sigma)
    return propagation_plots.plot_error_budget(budget).figure


def _amplification() -> plt.Figure:
    """Analytic, so it has no result file to wait for."""
    return propagation_plots.plot_amplification([0.0, 0.5, 0.9, 0.99]).figure


#: In report order. The section name is also the figures/ subdirectory.
FIGURES = [
    Figure("residual_histories", "convergence", _residual_histories,
           [(GRID, "history.csv")]),
    Figure("measured_vs_claimed_order", "convergence", _measured_vs_claimed_order,
           [(VERIFICATION, "summary.csv")]),
    Figure("order_vs_eccentricity", "convergence", _order_vs_eccentricity,
           [(VERIFICATION, "summary.csv")]),
    Figure("iterations_heatmaps", "grid", _iterations_heatmaps,
           [(GRID, "raw.csv")]),
    Figure("failure_maps", "grid", _failure_maps,
           [(GRID, "raw.csv")]),
    Figure("guess_effect", "grid", _guess_effect,
           [(GRID, "raw.csv")]),
    Figure("cost_breakdown", "cost", _cost_breakdown, []),
    Figure("cost_vs_accuracy", "cost", _cost_vs_accuracy,
           [(GRID, "raw.csv")]),
    Figure("wallclock_vs_cost", "cost", _wallclock_vs_cost,
           [(GRID, "timing.csv")]),
    Figure("tolerance_vs_parameter_shift", "propagation", _tolerance_vs_shift,
           [(PROPAGATION, "summary.csv")]),
    Figure("error_budget", "propagation", _error_budget,
           [(PROPAGATION, "summary.csv"), (PROPAGATION, "posterior_sigma.csv"),
            (MONTE_CARLO, "summary.csv")]),
    Figure("amplification", "propagation", _amplification, []),
]


# ----------------------------------------------------------------------
# Driving it
# ----------------------------------------------------------------------
def _drop_stale(figure: Figure) -> list[Path]:
    """Delete previous copies of a figure that did not rebuild."""
    removed = []
    for suffix in (".pdf", ".png"):
        path = FIGURE_DIR / figure.section / f"{figure.name}{suffix}"
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def _build(figure: Figure) -> tuple[str, str]:
    """Attempt one figure. Returns (status, detail)."""
    missing = figure.missing()
    if missing:
        return "missing input", f"needs {', '.join(missing)}"

    try:
        fig = figure.build()
    except NotImplementedError as exc:
        return "not written yet", str(exc)
    except Exception:
        return "failed", traceback.format_exc(limit=3).strip().splitlines()[-1]

    try:
        path = save_figure(fig, figure.name, subdir=figure.section)
    finally:
        plt.close(fig)
    return "ok", str(path.relative_to(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true",
                        help="show the plan and what each figure needs")
    args = parser.parse_args()

    if args.list:
        for figure in FIGURES:
            needs = ", ".join(f"{e}/{t}" for e, t in figure.needs) or "nothing"
            print(f"{figure.section}/{figure.name}  <- {needs}")
        return 0

    use_report_style()
    outcomes = []
    for figure in FIGURES:
        status, detail = _build(figure)
        outcomes.append((figure, status, detail))
        if status == "ok":
            print(f"  wrote {detail}")
        else:
            print(f"  {figure.section}/{figure.name}: {status} - {detail}")
            for path in _drop_stale(figure):
                print(f"    removed stale {path.relative_to(REPO_ROOT)}")

    built = [f for f, s, _ in outcomes if s == "ok"]
    print(f"\n{len(built)}/{len(FIGURES)} figures rebuilt into "
          f"{FIGURE_DIR.relative_to(REPO_ROOT)}/")
    if len(built) == len(FIGURES):
        return 0

    print("\nNot rebuilt:")
    for figure, status, detail in outcomes:
        if status != "ok":
            print(f"  {figure.section}/{figure.name:32s} {status}: {detail}")
    print("\nRun the missing experiments (scripts/run_*.py) and try again. "
          "The report must not be built from a partial set.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
