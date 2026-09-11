"""(e, M) grid figures. Owner: Mahdi."""

from __future__ import annotations


def plot_iterations_heatmap(df, solver: str, guess: str):
    """Heatmap of iterations to tolerance over the (e, M) plane.

    TODO(Mahdi): pcolormesh with e on one axis and M on the other, shared
    colour scale across all solvers so the panels are comparable. Mark
    non-converged cells in a distinct colour rather than leaving them blank.
    """
    raise NotImplementedError("plot_iterations_heatmap: see TODO above")


def plot_failure_map(df, solver: str):
    """Binary map of where a solver fails.

    TODO(Mahdi): most useful zoomed into the pathological corner with log
    axes in (1 - e) and M.
    """
    raise NotImplementedError("plot_failure_map: see TODO above")


def plot_guess_effect(df):
    """Iterations saved by each guess, per solver.

    TODO(Mahdi): grouped bars, solver on the x axis, one bar per guess,
    relative to the ``simple`` baseline. This is the figure that shows the
    guess layer was measured independently of the iteration.
    """
    raise NotImplementedError("plot_guess_effect: see TODO above")
