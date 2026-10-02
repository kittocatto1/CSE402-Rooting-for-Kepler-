"""Summary figures for the final report. Owner: Mahdi.

These are the grid-benchmark and safeguard figures the report actually
prints: the iteration heatmaps restricted to the even (uniform) block, the
per-solver failure maps, the mean-iteration bar charts by guess and by grid
set, and the safeguard failure counts.  ``grid_plots`` holds the general
per-panel drawing functions; this module arranges them into the specific
multi-panel layouts the report uses.

Like every plotting module here, these draw and never compute a metric
beyond grouping a result table and taking its mean.  Each function takes a
DataFrame read from ``results/`` and returns a matplotlib Figure.

Expected columns
----------------
``df``          one row per (solver, guess, grid point), as
                ``results/grid_benchmark/raw.csv``:
                solver, guess, e, M, converged, iterations
``summary_df``  one row per solver variant, as
                ``results/safeguard/summary.csv``:
                solver, n_failed_corner, fail_pct_corner, ...

Which set a grid point belongs to (uniform block, pathological corner or
RadVel sample) is recovered by matching its (e, M) against the grid
definitions in ``experiments.grid`` - see :func:`grid_set`.  It is never
re-typed here, so changing the grid config changes the figures with it.

Style note: these figures set their own look (which axis gets gridlines,
which spines are drawn) and are drawn under matplotlib's default rcParams
rather than ``style.use_report_style``.  The report style turns on gridlines
for every axes, which would draw lines across the heatmaps and a second set
of gridlines across the bar charts.
"""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from keplerbench.experiments import grid as grid_defs
from keplerbench.plotting import grid_plots

__all__ = [
    "grid_set",
    "plot_iterations_best_guess",
    "plot_failure_maps_by_solver",
    "plot_iterations_all_combinations",
    "plot_iterations_by_guess",
    "plot_iterations_by_set",
    "plot_safeguard_failures",
]

#: The iterative solvers, in the order every summary figure lists them.
ITERATIVE_SOLVERS = ("newton", "danby", "nwm9", "nwm11")

#: Every solver, for the failure maps (Markley is closed-form, so it has a
#: failure map but no guess comparison).
ALL_SOLVERS = ("markley", "danby", "newton", "nwm9", "nwm11")

#: The starting guesses, in the order every summary figure lists them.
GUESSES = ("simple", "canonical", "radvel", "napier")

#: The best starting guess for each solver, as chosen in the report.
BEST_GUESS_PAIRS = (("newton", "napier"), ("danby", "napier"),
                    ("nwm9", "napier"), ("nwm11", "canonical"))

#: One colour per guess: three blue-greys getting darker, orange for Napier.
GUESS_COLORS = ("#b8c4ce", "#7f99ad", "#4f7391", "#e07b24")

#: Colour of the "best value" labels in the per-set bar chart.
BEST_LABEL_COLOR = "#c25e0c"

#: Orange used for the guarded variant in the safeguard figure.
GUARDED_COLOR = "#e07b24"

#: Grid sets as (label from grid_set, panel title prefix).
SET_TITLES = (("uniform", "Even grid"), ("pathological", "Corner set"),
              ("radvel", "Radvel set"))

#: Tolerance for matching a sampled (e, M) to a grid-definition value.  The
#: sweep writes the values it was given, so they match to rounding error.
_MATCH_ATOL = 1e-12


def _matplotlib_defaults(build):
    """Draw ``build`` under matplotlib's default rcParams.

    See the module docstring for why these figures do not use the report
    style.  Only creation-time settings are affected; saving still follows
    whatever the caller has set (``save_figure`` and the report style).
    """
    @functools.wraps(build)
    def wrapper(*args, **kwargs):
        with mplstyle.context("default"):
            return build(*args, **kwargs)
    return wrapper


def solver_label(solver: str) -> str:
    """How a solver is named in figure text: NWM9, NWM11, Newton, Danby."""
    return solver.upper() if solver.startswith("nwm") else solver.capitalize()


def _on_axis(values: pd.Series, axis: np.ndarray) -> np.ndarray:
    """True where a value equals (to rounding) one of the axis values."""
    return np.isclose(values.to_numpy(dtype=float)[:, None], axis[None, :],
                      rtol=0.0, atol=_MATCH_ATOL).any(axis=1)


def _axes_of(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """The distinct e values and distinct M values of a lattice grid."""
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.unique(array[:, 0]), np.unique(array[:, 1])


def grid_set(df: pd.DataFrame, grid_spec: dict | None = None) -> pd.Series:
    """Label each row "uniform", "pathological" or "radvel".

    A point is in the uniform block when its e is one of the uniform grid's
    e values and its M one of its M values, and likewise for the
    pathological corner block; everything else is the RadVel sample, which
    is random and so cannot be matched by value.  The uniform test is
    applied last, so it wins if a point were ever on both lattices.

    ``grid_spec`` is a config's ``grid:`` block (for example
    ``load_config("configs/grid_benchmark.yaml").grid``).  Without one the
    defaults of ``experiments.grid`` are used, which are also what the
    benchmark config asks for.
    """
    spec = grid_spec or {}
    uniform_e, uniform_M = _axes_of(grid_defs.uniform_grid(**spec.get("uniform", {})))
    corner_e, corner_M = _axes_of(
        grid_defs.pathological_grid(**spec.get("pathological", {})))

    labels = pd.Series("radvel", index=df.index, name="set")
    labels[_on_axis(df["e"], corner_e) & _on_axis(df["M"], corner_M)] = "pathological"
    labels[_on_axis(df["e"], uniform_e) & _on_axis(df["M"], uniform_M)] = "uniform"
    return labels


def _uniform_block(df: pd.DataFrame, grid_spec: dict | None) -> pd.DataFrame:
    """The rows on the uniform lattice: a clean map with no stripes or scatter."""
    return df[grid_set(df, grid_spec) == "uniform"]


def _converged(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["converged"].astype(bool)]


# ----------------------------------------------------------------------
# 1. Heatmaps: the best guess per solver, and every combination
# ----------------------------------------------------------------------
@_matplotlib_defaults
def plot_iterations_best_guess(df: pd.DataFrame, grid_spec: dict | None = None,
                               pairs=BEST_GUESS_PAIRS) -> plt.Figure:
    """2 x 2 iteration heatmaps, each solver with its best starting guess.

    Drawn on the uniform 50 x 100 block only, so the map is a clean lattice
    without the corner stripes or the RadVel scatter.  One colour scale is
    shared by all four panels.
    """
    uniform = _uniform_block(df, grid_spec)
    vmax = grid_plots.shared_iteration_scale(uniform)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, (solver, guess) in zip(axes.ravel(), pairs):
        grid_plots.plot_iterations_heatmap(uniform, solver, guess, ax=ax, vmax=vmax)
        ax.set_title(f"{solver_label(solver)} + {guess}", fontsize=12)
    fig.tight_layout()
    return fig


@_matplotlib_defaults
def plot_iterations_all_combinations(df: pd.DataFrame,
                                     grid_spec: dict | None = None) -> plt.Figure:
    """Every (solver, guess) heatmap on the uniform block, plus Markley.

    Rows are solvers, columns guesses; Markley (no guess) sits alone in the
    top-right corner, with one shared colour bar below it.  The colour scale
    is the same as in :func:`plot_iterations_best_guess`.
    """
    uniform = _uniform_block(df, grid_spec)
    vmax = grid_plots.shared_iteration_scale(uniform)

    fig = plt.figure(figsize=(14, 7.4))
    gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1],
                  wspace=0.12, hspace=0.38)

    def panel(ax, solver, guess, title, left, bottom):
        grid_plots.plot_iterations_heatmap(uniform, solver, guess, ax=ax, vmax=vmax)
        mesh = ax.collections[0]
        # One shared colour bar for the figure instead of one per panel.
        mesh.colorbar.remove()
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("$M$" if bottom else "", fontsize=13)
        ax.set_ylabel("$e$" if left else "", fontsize=13)
        ax.tick_params(labelsize=10, labelleft=left, labelbottom=bottom)
        return mesh

    for i, solver in enumerate(ITERATIVE_SOLVERS):
        for j, guess in enumerate(GUESSES):
            mesh = panel(fig.add_subplot(gs[i, j]), solver, guess,
                         f"{solver_label(solver)} + {guess}",
                         j == 0, i == len(ITERATIVE_SOLVERS) - 1)
    mesh = panel(fig.add_subplot(gs[0, 4]), "markley",
                 grid_plots.GUESS_INDEPENDENT_LABEL, "Markley (no guess)",
                 False, True)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.06)
    box = gs[1:4, 4].get_position(fig)
    cax = fig.add_axes([box.x0 + 0.02, box.y0, 0.018, box.height])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("iterations to reach $10^{-14}$ (yellow = 5 or more)", fontsize=14)
    bar.ax.tick_params(labelsize=11)
    return fig


# ----------------------------------------------------------------------
# 2. Failure maps, one panel per solver
# ----------------------------------------------------------------------
@_matplotlib_defaults
def plot_failure_maps_by_solver(df: pd.DataFrame, solvers=ALL_SOLVERS) -> plt.Figure:
    """Where each solver failed, on a 2 x 3 grid (the sixth panel is hidden).

    Short titles and no per-panel failure counts in the legend; the counts
    are given in the report's tables instead.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    axes[1, 2].set_visible(False)
    for ax, solver in zip(axes.ravel(), solvers):
        grid_plots.plot_failure_map(df, solver, ax=ax)
        ax.set_title(solver_label(solver), fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [label.split(" (")[0] for label in labels],
                  loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 3. Mean iterations by guess, overall and per grid set
# ----------------------------------------------------------------------
def _is_best(value: float, candidates) -> bool:
    """Bold a bar when it ties the best (lowest) value at one decimal."""
    return round(value, 1) == min(round(v, 1) for v in candidates)


@_matplotlib_defaults
def plot_iterations_by_guess(df: pd.DataFrame) -> plt.Figure:
    """Mean iterations for each solver under each guess, over the whole grid.

    Only converged solves are averaged.  A fifth, hatched bar per solver is
    its mean over all four guesses together.  The best guess for each
    solver has its value in bold.
    """
    converged = _converged(df)
    mean = converged.groupby(["solver", "guess"])["iterations"].mean()
    solver_mean = converged.groupby("solver")["iterations"].mean()

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    width = 0.17
    x = np.arange(len(ITERATIVE_SOLVERS))
    for j, (guess, color) in enumerate(zip(GUESSES, GUESS_COLORS)):
        values = [mean[(s, guess)] for s in ITERATIVE_SOLVERS]
        bars = ax.bar(x + (j - 2) * width, values, width * 0.92,
                      label=guess, color=color)
        for bar, value, solver in zip(bars, values, ITERATIVE_SOLVERS):
            best = _is_best(value, [mean[(solver, g)] for g in GUESSES])
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if best else "normal")

    values = [solver_mean[s] for s in ITERATIVE_SOLVERS]
    bars = ax.bar(x + 2 * width + 0.03, values, width * 0.92,
                  label="mean of all 4", color="white", edgecolor="#222",
                  hatch="///", lw=1.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05,
                f"{value:.2f}", ha="center", va="bottom", fontsize=10,
                color="#222", style="italic")

    ax.set_xticks(x + 0.02)
    ax.set_xticklabels([solver_label(s) for s in ITERATIVE_SOLVERS], fontsize=14)
    ax.set_ylabel("mean iterations to reach $10^{-14}$", fontsize=13)
    ax.set_ylim(0, 4.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title="starting guess", ncol=2, loc="upper right", fontsize=11.5,
              title_fontsize=12, frameon=False)
    fig.tight_layout()
    return fig


@_matplotlib_defaults
def plot_iterations_by_set(df: pd.DataFrame,
                           grid_spec: dict | None = None) -> plt.Figure:
    """Mean iterations per solver and guess, split by grid set.

    Three panels of horizontal bars: the even (uniform) grid, the corner
    set and the RadVel set, each titled with its number of grid points.
    Each solver's tick label carries its mean over all four guesses, and the
    best guess for each solver is labelled in bold orange.
    """
    df = df.assign(set=grid_set(df, grid_spec))
    converged = _converged(df)
    mean = converged.groupby(["set", "solver", "guess"])["iterations"].mean()
    n_points = df.drop_duplicates(["e", "M"]).groupby("set").size()

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.5), sharex=True)
    height = 0.2
    y = np.arange(len(ITERATIVE_SOLVERS))
    for ax, (set_name, title) in zip(axes, SET_TITLES):
        solver_mean = (converged[converged["set"] == set_name]
                       .groupby("solver")["iterations"].mean())
        for j, (guess, color) in enumerate(zip(GUESSES, GUESS_COLORS)):
            values = [mean[(set_name, s, guess)] for s in ITERATIVE_SOLVERS]
            bars = ax.barh(y + (j - 1.5) * height, values, height * 0.9,
                           label=guess, color=color)
            for bar, value, solver in zip(bars, values, ITERATIVE_SOLVERS):
                best = _is_best(value, [mean[(set_name, solver, g)] for g in GUESSES])
                ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}", va="center", fontsize=10.5,
                        fontweight="bold" if best else "normal",
                        color=BEST_LABEL_COLOR if best else "#333")
        ax.set_title(f"{title} ({int(n_points.get(set_name, 0)):,})",
                     fontsize=13, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{solver_label(s)}\nmean {solver_mean[s]:.2f}"
                            for s in ITERATIVE_SOLVERS],
                           fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("mean iterations", fontsize=12)
    axes[0].set_xlim(0, 8.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="starting guess", ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=12, title_fontsize=12,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.2)
    return fig


# ----------------------------------------------------------------------
# 4. What the safeguard buys in the hard corner
# ----------------------------------------------------------------------
#: The three NWM variants in the safeguard experiment, as
#: (suffix on the solver name, legend label, colour).
SAFEGUARD_VARIANTS = (
    ("", "published", "#4f7391"),
    ("_memoryless", "memory off", "#b8c4ce"),
    ("_guarded", "guarded", GUARDED_COLOR),
)

#: Colour of the single Danby reference bar.
REFERENCE_COLOR = "#8c8c8c"


def _corner_total(summary: pd.DataFrame) -> int | None:
    """How many solves the hard corner holds, recovered from the summary.

    ``fail_pct_corner`` is ``100 * n_failed_corner / n_corner``, so any row
    with at least one corner failure gives the denominator back.  None when
    no row has one (then the axis label simply omits the count).
    """
    rows = summary[summary["n_failed_corner"] > 0]
    if rows.empty or "fail_pct_corner" not in summary.columns:
        return None
    row = rows.iloc[0]
    return int(round(100.0 * row["n_failed_corner"] / row["fail_pct_corner"]))


@_matplotlib_defaults
def plot_safeguard_failures(summary_df: pd.DataFrame,
                            solvers=("nwm9", "nwm11"),
                            reference: str = "danby") -> plt.Figure:
    """Hard-corner failures for each NWM variant, with Danby for reference.

    For each NWM solver three bars: the published method, the same method
    with its memory switched off, and the guarded version.  A single grey
    bar on the right shows the reference solver.  Every bar is labelled with
    its count.  No title; the caption lives in the report.
    """
    missing = [c for c in ("solver", "n_failed_corner") if c not in summary_df.columns]
    if missing:
        raise KeyError(f"plot_safeguard_failures: DataFrame is missing {missing}; "
                       f"got {list(summary_df.columns)}")
    counts = summary_df.set_index("solver")["n_failed_corner"]
    wanted = [f"{s}{suffix}" for s in solvers for suffix, _, _ in SAFEGUARD_VARIANTS]
    wanted.append(reference)
    missing = [name for name in wanted if name not in counts.index]
    if missing:
        raise ValueError(f"plot_safeguard_failures: no rows for {missing}; "
                         f"got {sorted(counts.index)}")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    width = 0.26
    centres = np.arange(len(solvers), dtype=float)
    top = float(counts[wanted].max())

    def label(bar, value):
        ax.text(bar.get_x() + bar.get_width() / 2, value + top * 0.015,
                f"{int(value)}", ha="center", va="bottom", fontsize=12)

    for k, (suffix, name, color) in enumerate(SAFEGUARD_VARIANTS):
        values = [float(counts[f"{s}{suffix}"]) for s in solvers]
        bars = ax.bar(centres + (k - 1) * width, values, width * 0.92,
                      label=name, color=color)
        for bar, value in zip(bars, values):
            label(bar, value)

    ref_x = len(solvers) - 0.25
    ref_value = float(counts[reference])
    (ref_bar,) = ax.bar([ref_x], [ref_value], width * 0.92, color=REFERENCE_COLOR)
    label(ref_bar, ref_value)

    ax.set_xticks(list(centres) + [ref_x])
    ax.set_xticklabels([solver_label(s) for s in solvers] + [solver_label(reference)],
                       fontsize=13)
    total = _corner_total(summary_df)
    ylabel = "failed solves in the hard corner"
    if total is not None:
        ylabel += f" (of {total:,})"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, top * 1.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    fig.tight_layout()
    return fig
