"""Convergence figures. Owner: Suchi."""

from __future__ import annotations


def plot_residual_history(history_df, e: float, M: float):
    """log10|f(E_n)| vs iteration, one line per solver, at one (e, M).

    TODO(Suchi): semilogy on the residual; mark the double-precision floor
    with a horizontal line so readers see when a curve stops being
    meaningful. Pick one "typical" and one "hard corner" (e, M) and show
    both side by side - the story is different in each.
    """
    raise NotImplementedError("plot_residual_history: see TODO above")


def plot_measured_vs_claimed_order(order_df):
    """Bar chart: measured empirical order next to the paper's claim.

    TODO(Suchi): one pair of bars per solver, with error bars from the
    spread across test points. Where the measurement was unreliable (too few
    usable iterations), draw the bar hatched rather than omitting it, and
    explain in the caption.
    """
    raise NotImplementedError("plot_measured_vs_claimed_order: see TODO above")


def plot_order_vs_eccentricity(order_df):
    """Measured order as a function of e.

    TODO(Suchi): tests whether the with-memory methods hold their claimed
    order across the whole range or degrade as e -> 1. This is one of the
    project's central questions, so it deserves its own figure.
    """
    raise NotImplementedError("plot_order_vs_eccentricity: see TODO above")
