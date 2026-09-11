"""Cost-accounting figures. Owner: Dipit."""

from __future__ import annotations


def plot_cost_breakdown(cost_table_df):
    """Stacked bars: per-iteration cost split by evaluation kind.

    TODO(Dipit): one bar per solver, stacked into sincos pairs / sin only /
    cos only / synthesised. This makes the Section 4.1 argument visible in
    one picture and belongs early in the results section.
    """
    raise NotImplementedError("plot_cost_breakdown: see TODO above")


def plot_cost_vs_accuracy(df):
    """Weighted cost against digits of accuracy achieved - the money plot.

    TODO(Dipit): scatter/line per solver, cost on x and correct digits on y.
    A method that reaches the same accuracy further left is better, and this
    is the fairest way to compare methods of different order. Add the
    efficiency-index ranking as a second panel so the reader can see where
    the generic index and the Kepler-specific costing disagree.
    """
    raise NotImplementedError("plot_cost_vs_accuracy: see TODO above")


def plot_wallclock_vs_cost(df):
    """Measured wall-clock time against the weighted cost model.

    TODO(Dipit): if the cost model is any good these should correlate. If
    they do not, the weights are wrong or Python overhead dominates - either
    way it must be reported, not hidden.
    """
    raise NotImplementedError("plot_wallclock_vs_cost: see TODO above")
