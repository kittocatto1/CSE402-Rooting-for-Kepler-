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

_PARAM_LABELS = {"P": "$P$", "e": "$e$", "omega": r"$\omega$", "K": "$K$",
                "gamma": r"$\gamma$", "jitter": "jitter"}


def _require(df, columns: list[str], who: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{who}: DataFrame is missing {missing}; got {list(df.columns)}")


def _solver_color(name: str) -> str:
    return SOLVER_COLORS.get(str(name), "#666666")


def plot_tolerance_vs_parameter_shift(shift_df):
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
            # exact zeros can't sit on a log axis: draw them hollow at a floor
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
        ax.invert_xaxis()
        ax.set_xlabel("solver tolerance")
        ax.set_ylabel(f"|shift| in {_PARAM_LABELS.get(param, param)}  ($\\sigma$)")
        ax.set_title(_PARAM_LABELS.get(param, param))

    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_error_budget(budget_df, ax=None):
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


def plot_amplification(e_values, ax=None):
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
