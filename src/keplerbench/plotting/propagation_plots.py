"""Downstream propagation figures. Owner: Fariha."""

from __future__ import annotations


def plot_tolerance_vs_parameter_shift(shift_df):
    """Solver tolerance on x, shift in P / e / K on y, in units of sigma.

    TODO(Fariha): one panel per parameter, one line per solver, log x axis.
    Draw a horizontal line at 1 sigma (and maybe 0.1 sigma) so the reader can
    read off the tolerance at which the solver stops mattering.
    """
    raise NotImplementedError("plot_tolerance_vs_parameter_shift: see TODO above")


def plot_error_budget(budget_df):
    """Side-by-side bars: solver-induced shift vs noise spread vs MCMC sigma.

    TODO(Fariha): one group per parameter. This is the figure that answers
    the project's third expected outcome, so make it readable on a slide.
    """
    raise NotImplementedError("plot_error_budget: see TODO above")


def plot_amplification(e_values):
    """dnu/dE against E for several eccentricities.

    TODO(Fariha): analytic curve from rv.anomaly.dE_to_dnu. Explains WHY the
    high-eccentricity corner matters downstream, and costs nothing to make.
    """
    raise NotImplementedError("plot_amplification: see TODO above")
