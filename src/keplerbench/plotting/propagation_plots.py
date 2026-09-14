"""Downstream propagation figures. Owner: Fariha.

These draw, they never compute - every function takes a DataFrame (or dict)
that ``evaluation/propagation.py`` produced, matching the convention the
rest of ``plotting/`` follows.

Expected inputs
---------------
``shift_df``    one row per (solver, tolerance), as produced by
                ``evaluation.propagation.shift_in_sigma``:
                solver, tolerance, <param>_shift, <param>_shift_sigma (+
                <param>_shift_rel), for param in e/omega/K/gamma/jitter.

``budget_df``   one row per parameter, as produced by
                ``evaluation.propagation.compare_error_budgets``:
                parameter, solver_induced_shift, noise_driven_spread,
                mcmc_sigma, ratio_solver_to_noise.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from keplerbench.evaluation.propagation import TRACKED_PARAMS
from keplerbench.plotting.style import SOLVER_COLORS
from keplerbench.rv.anomaly import dE_to_dnu

__all__ = [
    "plot_tolerance_vs_parameter_shift",
    "plot_error_budget",
    "plot_amplification",
]

#: Pretty labels for the report/slides.
_PARAM_LABELS = {"e": "$e$", "omega": r"$\omega$", "K": "$K$",
                "gamma": r"$\gamma$", "jitter": "jitter"}


def _require(df, columns: list[str], who: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{who}: DataFrame is missing {missing}; got {list(df.columns)}")


def _solver_color(name: str) -> str:
    return SOLVER_COLORS.get(str(name), "#666666")


# ----------------------------------------------------------------------
# 1. Tolerance vs parameter shift
# ----------------------------------------------------------------------
def plot_tolerance_vs_parameter_shift(shift_df):
    """Solver tolerance on x, shift in each parameter on y, in units of sigma.

    One panel per parameter that has a ``<param>_shift_sigma`` column, one
    line per solver, log x axis. Horizontal reference lines at 1 sigma and
    0.1 sigma let the reader read off the tolerance at which the solver
    stops mattering.

    A shift of EXACTLY zero (two tolerances landing on a bit-identical fit -
    e.g. once solver tolerance is tighter than the optimiser's own xtol,
    RadVel's Powell fit stops moving at all) cannot be drawn on a log axis.
    Those points are drawn as hollow markers at a shared floor rather than
    silently dropped, the same convention ``convergence_plots.py`` uses for
    exact-zero residuals: "hit zero" is a real, different outcome from
    "very small", and dropping the point would look like missing data.
    """
    _require(shift_df, ["solver", "tolerance"], "plot_tolerance_vs_parameter_shift")

    params = [p for p in TRACKED_PARAMS if f"{p}_shift_sigma" in shift_df.columns]
    if not params:
        raise KeyError(
            "plot_tolerance_vs_parameter_shift: no <param>_shift_sigma column "
            f"found; got {list(shift_df.columns)}"
        )

    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.6),
                             squeeze=False, sharex=True)
    axes = axes[0]

    for ax, param in zip(axes, params):
        col = f"{param}_shift_sigma"
        abs_shift = shift_df[col].abs()
        positive = abs_shift[abs_shift > 0]
        floor = positive.min() * 0.5 if not positive.empty else 1e-12

        for solver, group in shift_df.groupby("solver", sort=False):
            group = group.sort_values("tolerance")
            tol = group["tolerance"].to_numpy(dtype=float)
            value = group[col].abs().to_numpy(dtype=float)
            is_positive = value > 0

            ax.plot(tol[is_positive], value[is_positive], color=_solver_color(solver),
                   marker="o", markersize=3.5, linewidth=1.6, label=str(solver))
            if (~is_positive).any():
                ax.plot(tol[~is_positive], np.full((~is_positive).sum(), floor),
                       color=_solver_color(solver), linestyle="none",
                       marker="o", markersize=6, markerfacecolor="none")

        ax.axhspan(0, floor, color="0.9", zorder=0)
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
        ax.axhline(0.1, color="0.6", linestyle=":", linewidth=1.0)
        ax.annotate("1$\\sigma$", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                   fontsize=7, va="bottom", color="0.35")
        ax.annotate("0.1$\\sigma$", xy=(0.02, 0.1), xycoords=("axes fraction", "data"),
                   fontsize=7, va="bottom", color="0.5")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()  # tighter tolerance (smaller number) reads left-to-right as "more precise"
        ax.set_xlabel("solver tolerance")
        ax.set_ylabel(f"|shift| in {_PARAM_LABELS.get(param, param)}  ($\\sigma$)")
        ax.set_title(_PARAM_LABELS.get(param, param))

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. Error budget bars
# ----------------------------------------------------------------------
def plot_error_budget(budget_df, ax=None):
    """Side-by-side bars: solver-induced shift vs noise spread vs MCMC sigma.

    One group per parameter, log y axis - the whole point of this figure is
    that these three quantities live at very different scales, and the plot
    should make that gap impossible to miss.
    """
    _require(budget_df, ["parameter", "solver_induced_shift",
                        "noise_driven_spread", "mcmc_sigma"], "plot_error_budget")

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(1.6 * len(budget_df) + 2, 4.0))

    series = [
        ("solver-induced shift", budget_df["solver_induced_shift"], "#4C72B0"),
        ("noise-driven spread", budget_df["noise_driven_spread"], "#DD8452"),
        ("MCMC sigma", budget_df["mcmc_sigma"], "#55A868"),
    ]
    n_groups = len(budget_df)
    n_bars = len(series)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, (label, values, color) in enumerate(series):
        offset = (i - (n_bars - 1) / 2) * width
        values = values.to_numpy(dtype=float)
        plottable = np.where(values > 0, values, np.nan)
        ax.bar(x + offset, plottable, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([_PARAM_LABELS.get(p, p) for p in budget_df["parameter"]])
    ax.set_yscale("log")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_title("Error budget: solver choice vs. everything else")
    ax.legend(fontsize=8)
    return ax.figure if created else ax


# ----------------------------------------------------------------------
# 3. Analytic amplification factor
# ----------------------------------------------------------------------
def plot_amplification(e_values, ax=None):
    """dnu/dE against E for several eccentricities.

    Purely analytic (``rv.anomaly.dE_to_dnu``) - explains WHY the
    high-eccentricity corner matters downstream: an error in E is amplified
    by exactly this factor on its way into nu, and it grows sharply for
    e -> 1 near E -> 0.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots()

    E = np.linspace(0.0, 2.0 * np.pi, 400)
    for e in e_values:
        amplification = [dE_to_dnu(float(Ei), float(e)) for Ei in E]
        ax.plot(E, amplification, label=f"$e={e:g}$", linewidth=1.6)

    ax.set_xlabel("eccentric anomaly $E$")
    ax.set_ylabel(r"$d\nu/dE$")
    ax.set_title("Error amplification from $E$ into $\\nu$")
    ax.legend(fontsize=8)
    return ax.figure if created else ax
