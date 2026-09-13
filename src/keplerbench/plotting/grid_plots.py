"""(e, M) grid figures. Owner: Mahdi.

These draw, they never compute.  Every function takes a DataFrame that
``experiments/`` or ``evaluation/`` produced and turns it into a figure, so
any figure in the report can be traced back to a result file.

Expected columns
----------------
``df``  one row per (solver, guess, grid point), as written by
        ``io.results_io.save_results``:
        solver, guess, e, M, converged, iterations  (+ residual, error,
        failure)

The pathological corner is never redefined here.  ``HARD_CORNER_E`` and
``HARD_CORNER_M`` are imported from ``evaluation.robustness`` so the shaded
region in a figure and the row labelled ``hard_corner`` in a table always
mean the same thing.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keplerbench.evaluation.robustness import HARD_CORNER_E, HARD_CORNER_M
from keplerbench.plotting.style import SOLVER_COLORS

__all__ = [
    "plot_iterations_heatmap",
    "plot_failure_map",
    "plot_guess_effect",
]

#: Colour for cells the solver never converged on.  Deliberately outside the
#: sequential colormap's range so a failure can never be misread as "just a
#: lot of iterations" - those are different outcomes and the report must not
#: blur them.
NON_CONVERGED_COLOR = "#B0B0B0"

#: Colour for (e, M) cells the grid never sampled.  Distinct from
#: NON_CONVERGED_COLOR: "not measured" and "measured, failed" are different
#: claims.
MISSING_COLOR = "#FFFFFF"

#: The closed-form solvers are run once under this pseudo-guess label, so a
#: guess-vs-guess comparison has nothing to say about them.  See
#: ``experiments.runner.GUESS_INDEPENDENT``.
GUESS_INDEPENDENT_LABEL = "n/a"

#: Baseline every other guess is measured against in plot_guess_effect.
BASELINE_GUESS = "simple"


def _require(df: pd.DataFrame, columns: list[str], who: str) -> None:
    """Fail loudly on a missing column.

    A silently empty figure in the report is worse than a crash here.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{who}: DataFrame is missing {missing}; got {list(df.columns)}"
        )


def _solver_color(name: str) -> str:
    return SOLVER_COLORS.get(str(name), "#666666")


def _select(df: pd.DataFrame, who: str, **equals: object) -> pd.DataFrame:
    """Filter on exact column values, failing loudly when nothing matches.

    An empty selection means the caller asked for a solver/guess combination
    the sweep never ran - drawing an empty panel would hide that.
    """
    selection = df
    for column, value in equals.items():
        selection = selection[selection[column] == value]
    if selection.empty:
        asked = ", ".join(f"{k}={v!r}" for k, v in equals.items())
        raise ValueError(f"{who}: no rows for {asked}")
    return selection


def _grid_axes(selection: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The sorted unique e and M values this selection was sampled on."""
    return (np.sort(selection["e"].unique()),
            np.sort(selection["M"].unique()))


def _pivot(selection: pd.DataFrame, values: str,
           e_values: np.ndarray, M_values: np.ndarray) -> np.ndarray:
    """(e, M) -> value as a 2-D array with NaN wherever nothing was sampled.

    The grids are unions of a uniform block, a logarithmic corner block and a
    random RadVel sample, so the (e, M) points are emphatically *not* a full
    rectangular lattice.  Reindexing onto one leaves genuine holes, and those
    holes must stay NaN rather than being filled with a neighbour.
    """
    table = selection.pivot_table(index="e", columns="M", values=values,
                                  aggfunc="mean")
    return table.reindex(index=e_values, columns=M_values).to_numpy(dtype=float)


def _mesh_edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges around a set of (possibly very unevenly spaced) centres."""
    centres = np.asarray(centres, dtype=float)
    if centres.size == 1:
        width = max(abs(centres[0]) * 0.01, 1e-9)
        return np.array([centres[0] - width, centres[0] + width])
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    first = centres[0] - (midpoints[0] - centres[0])
    last = centres[-1] + (centres[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


# ----------------------------------------------------------------------
# 1. Iteration count over the (e, M) plane
# ----------------------------------------------------------------------
def plot_iterations_heatmap(df, solver: str, guess: str, ax=None,
                            vmax: float | None = None, cmap: str = "viridis"):
    """Iterations to tolerance over the (e, M) plane for one (solver, guess).

    ``vmax`` fixes the top of the colour scale.  Pass the *same* ``vmax`` to
    every panel in a figure - otherwise each panel is normalised to its own
    worst cell and two panels that look identical are describing different
    iteration counts.  ``shared_iteration_scale`` computes a suitable value
    across the whole sweep.

    Non-converged cells are drawn in a flat grey rather than at the top of
    the colormap, because "did not converge" is not "converged, slowly".
    """
    _require(df, ["solver", "guess", "e", "M", "iterations", "converged"],
             "plot_iterations_heatmap")
    selection = _select(df, "plot_iterations_heatmap",
                        solver=solver, guess=guess)

    e_values, M_values = _grid_axes(selection)
    converged = selection[selection["converged"].astype(bool)]

    iterations = (_pivot(converged, "iterations", e_values, M_values)
                  if not converged.empty
                  else np.full((e_values.size, M_values.size), np.nan))
    # 1.0 marks a sampled-but-failed cell, NaN an unsampled one.
    sampled = _pivot(selection, "converged", e_values, M_values)
    failed = np.where(np.isnan(sampled), np.nan, 1.0 - sampled)

    if ax is None:
        _, ax = plt.subplots()

    colormap = plt.get_cmap(cmap).with_extremes(bad=MISSING_COLOR)
    mesh = ax.pcolormesh(
        _mesh_edges(M_values), _mesh_edges(e_values),
        np.ma.masked_invalid(iterations),
        cmap=colormap, vmin=0.0, vmax=vmax, shading="flat",
    )
    # Overlay the failures so they sit on top of the colormap, never in it.
    failure_layer = np.ma.masked_where(~(failed > 0.0), failed)
    if failure_layer.count():
        from matplotlib.colors import ListedColormap
        ax.pcolormesh(
            _mesh_edges(M_values), _mesh_edges(e_values), failure_layer,
            cmap=ListedColormap([NON_CONVERGED_COLOR]), shading="flat",
        )

    ax.set_xlabel("mean anomaly $M$")
    ax.set_ylabel("eccentricity $e$")
    ax.set_title(f"{solver} + {guess}")
    ax.grid(False)
    colorbar = ax.figure.colorbar(mesh, ax=ax)
    colorbar.set_label("iterations to tolerance")
    return ax.figure


def shared_iteration_scale(df, percentile: float = 99.0) -> float:
    """A ``vmax`` that makes every panel of a heatmap figure comparable.

    Taken from a high percentile rather than the maximum: one pathological
    cell at 50 iterations would otherwise compress every other panel into the
    bottom of the colour scale and hide the differences the figure exists to
    show.
    """
    _require(df, ["iterations", "converged"], "shared_iteration_scale")
    converged = df[df["converged"].astype(bool)]
    if converged.empty:
        raise ValueError("shared_iteration_scale: no converged rows")
    return float(np.percentile(converged["iterations"], percentile))


# ----------------------------------------------------------------------
# 2. Where a solver fails
# ----------------------------------------------------------------------
def plot_failure_map(df, solver: str, ax=None, guess: str | None = None):
    """Where ``solver`` failed, on log axes in (1 - e) and M.

    Linear axes put almost every sampled point in one corner of the frame;
    the pathological region is only legible in ``log10(1 - e)`` against
    ``log10 M``, which is also how ``grid.pathological_grid`` samples it.

    Converged and failed points are drawn as separate scatter layers rather
    than as a binary image, because the grid is not a rectangular lattice
    (see ``_pivot``) and a pcolormesh would invent cells that were never run.
    """
    _require(df, ["solver", "guess", "e", "M", "converged"],
             "plot_failure_map")
    equals: dict[str, object] = {"solver": solver}
    if guess is not None:
        equals["guess"] = guess
    selection = _select(df, "plot_failure_map", **equals)

    # A point counts as failed if ANY guess failed there, so the map answers
    # "can this solver be trusted here" rather than "does one lucky guess
    # rescue it". With guess= supplied it is that guess alone.
    per_point = (selection.groupby(["e", "M"])["converged"]
                 .apply(lambda s: bool(s.astype(bool).all()))
                 .reset_index())
    one_minus_e = 1.0 - per_point["e"].to_numpy(dtype=float)
    M = per_point["M"].to_numpy(dtype=float)
    ok = per_point["converged"].to_numpy(dtype=bool)

    # log axes cannot show e == 0 or M == 0; drop them and say how many.
    drawable = (one_minus_e > 0.0) & (M > 0.0)
    n_dropped = int((~drawable).sum())

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(M[drawable & ok], one_minus_e[drawable & ok],
               s=12, c=_solver_color(solver), alpha=0.45,
               edgecolors="none", label="converged")
    failed = drawable & ~ok
    ax.scatter(M[failed], one_minus_e[failed],
               s=28, c="none", edgecolors="#B22222", linewidths=1.1,
               label=f"failed ({int(failed.sum())})")

    ax.axhline(1.0 - HARD_CORNER_E, color="#444444", lw=0.8, ls="--")
    ax.axvline(HARD_CORNER_M, color="#444444", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()  # e -> 1 (the hard limit) at the top
    ax.set_xlabel("mean anomaly $M$")
    ax.set_ylabel(r"$1 - e$")
    title = solver if guess is None else f"{solver} + {guess}"
    if n_dropped:
        title += f"  ({n_dropped} points at e=0 or M=0 not drawable on log axes)"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax.figure


# ----------------------------------------------------------------------
# 3. What the guess layer is worth
# ----------------------------------------------------------------------
def plot_guess_effect(df, ax=None, baseline: str = BASELINE_GUESS):
    """Iterations saved by each guess relative to ``simple``, per solver.

    This is the figure that shows the guess layer was measured independently
    of the iteration: a bar is a property of the *starting point*, at a fixed
    solver, so a tall bar cannot be confused with a better update rule.

    Only rows that converged under BOTH the baseline and the compared guess
    enter a bar.  Averaging over whatever happened to converge would let a
    guess look good precisely because it failed on the hard points and so
    never paid for them.

    Closed-form solvers are excluded - they ignore E0 entirely and are run
    once under the "n/a" label, so they have no guess effect to show.
    """
    _require(df, ["solver", "guess", "e", "M", "iterations", "converged"],
             "plot_guess_effect")

    usable = df[df["guess"] != GUESS_INDEPENDENT_LABEL]
    if usable.empty:
        raise ValueError(
            "plot_guess_effect: every row is guess-independent; "
            "there is no guess effect to draw"
        )
    if baseline not in set(usable["guess"]):
        raise ValueError(
            f"plot_guess_effect: baseline guess {baseline!r} is not in the "
            f"data; got {sorted(set(usable['guess']))}"
        )

    converged = usable[usable["converged"].astype(bool)]
    base = (converged[converged["guess"] == baseline]
            .set_index(["solver", "e", "M"])["iterations"])

    other_guesses = [g for g in sorted(set(usable["guess"])) if g != baseline]
    solvers = sorted(set(usable["solver"]))

    if ax is None:
        _, ax = plt.subplots()

    width = 0.8 / max(len(other_guesses), 1)
    positions = np.arange(len(solvers), dtype=float)
    hatches = ["", "//", "\\\\", "xx", ".."]

    for k, guess in enumerate(other_guesses):
        rows = (converged[converged["guess"] == guess]
                .set_index(["solver", "e", "M"])["iterations"])
        paired = pd.concat([base.rename("base"), rows.rename("this")],
                           axis=1, join="inner")
        saved = (paired["base"] - paired["this"]).groupby(level="solver").mean()
        heights = [saved.get(s, np.nan) for s in solvers]
        ax.bar(positions + (k - (len(other_guesses) - 1) / 2) * width, heights,
               width=width, label=guess, hatch=hatches[k % len(hatches)],
               edgecolor="#333333", linewidth=0.6,
               color=[_solver_color(s) for s in solvers])

    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(solvers)
    ax.set_ylabel(f"mean iterations saved vs. {baseline}")
    ax.set_xlabel("solver")
    ax.set_title(f"Effect of the starting guess (baseline: {baseline})")
    ax.legend(loc="best", fontsize=8, title="guess")
    return ax.figure
