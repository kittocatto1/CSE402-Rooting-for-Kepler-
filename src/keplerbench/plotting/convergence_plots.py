"""Convergence figures. Owner: Suchi.

These functions only draw. They never work anything out. Each one takes a
table that ``evaluation/`` or ``experiments/`` already produced and turns it
into a figure, so every figure in the report can be traced back to a file.

What the tables must contain
----------------------------
``history_df``  one row per iteration of each solve, as written by
                ``io.results_io.save_history``:
                solver, guess, e, M, iteration, residual  (+ error, step)

``order_df``    one row per measurement, from
                ``experiments.verification.verify_order_on_test_functions``
                or from an order table over the (e, M) grid:
                solver, measured_order, claimed_order  (+ function, e,
                n_usable)
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keplerbench.plotting.style import GUESS_STYLES, SOLVER_COLORS

__all__ = [
    "plot_residual_history",
    "plot_residual_history_pair",
    "plot_measured_vs_claimed_order",
    "plot_order_vs_eccentricity",
]

#: With fewer usable terms than this, the estimator never saw the method
#: settle into its true rate, so its answer is not really a measurement.
#: Those bars and markers are drawn hatched or hollow rather than left out.
#: A missing bar says "no data", which is a different and more misleading
#: claim than "measured, but do not trust it".
MIN_USABLE_TERMS = 3

#: How small a gap a double can represent. Used to place the noise floor.
DOUBLE_EPS = 2.220446049250313e-16


def _require(df: pd.DataFrame, columns: list[str], who: str) -> None:
    """Stop with a clear error if a column is missing.

    A blank figure that reaches the report is worse than a crash here.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{who}: DataFrame is missing {missing}; got {list(df.columns)}"
        )


def _solver_color(name: str) -> str:
    return SOLVER_COLORS.get(str(name), "#666666")


def _guess_style(name: str) -> str:
    return GUESS_STYLES.get(str(name), "-")


# ----------------------------------------------------------------------
# 1. Residual histories
# ----------------------------------------------------------------------
def plot_residual_history(history_df, e: float, M: float, ax=None, atol=1e-12):
    """Residual against iteration number, one line per solver, at one (e, M).

    The shaded band is the noise floor. We compute f(E) as E - e sin E - M,
    so its own rounding error is roughly eps * max(|E|, |M|). Below that
    line a curve is showing rounding noise, not convergence, and the report
    should read nothing into how far it drops.

    A residual of exactly zero cannot go on a log axis. Those are drawn at
    the floor with a hollow marker and named in the legend, because landing
    exactly on the root is a real outcome and a different one from stalling.
    """
    _require(history_df, ["solver", "e", "M", "iteration", "residual"],
             "plot_residual_history")

    selection = history_df[
        np.isclose(history_df["e"], e, rtol=0, atol=atol)
        & np.isclose(history_df["M"], M, rtol=0, atol=atol)
    ]
    if selection.empty:
        raise ValueError(
            f"no rows for e={e!r}, M={M!r}; the history table has "
            f"{len(history_df)} rows covering "
            f"e in [{history_df['e'].min()}, {history_df['e'].max()}]"
        )

    created = ax is None
    if created:
        _, ax = plt.subplots()

    has_guesses = "guess" in selection.columns and selection["guess"].nunique() > 1
    keys = ["solver", "guess"] if has_guesses else ["solver"]

    floor = DOUBLE_EPS * max(abs(float(selection["E"].abs().max()))
                             if "E" in selection.columns else 1.0,
                             abs(float(M)), 1.0)

    for key, group in selection.groupby(keys, sort=True):
        # groupby on a one-item list gives back a 1-tuple, not a plain value.
        key = key if isinstance(key, tuple) else (key,)
        solver = key[0]
        guess = key[1] if len(key) > 1 else None
        group = group.sort_values("iteration")

        residual = group["residual"].to_numpy(dtype=float)
        iteration = group["iteration"].to_numpy(dtype=float)
        positive = residual > 0

        label = f"{solver}" + (f" + {guess}" if guess else "")
        ax.semilogy(
            iteration[positive], residual[positive],
            color=_solver_color(solver),
            linestyle=_guess_style(guess) if guess else "-",
            marker="o", markersize=3.5, linewidth=1.6, label=label,
        )
        if (~positive).any():
            ax.semilogy(
                iteration[~positive], np.full((~positive).sum(), floor),
                color=_solver_color(solver), linestyle="none",
                marker="o", markersize=6, markerfacecolor="none",
                label=f"{label} (exact zero)",
            )

    ax.axhspan(0, floor, color="0.85", zorder=0)
    ax.axhline(floor, color="0.45", linestyle="--", linewidth=1.0, zorder=1)
    ax.annotate("double-precision floor", xy=(0.99, floor), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="0.35")

    ax.set_xlabel("iteration $n$")
    ax.set_ylabel(r"residual $|f(E_n)|$")
    ax.set_title(f"$e = {e:g}$,  $M = {M:g}$", fontsize=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    return ax.figure if created else ax


def plot_residual_history_pair(history_df, typical, hard):
    """Two residual histories side by side: an ordinary case and a hard one.

    The two really do behave differently. In the ordinary range the fast
    methods reach the floor in two or three steps; in the hard corner
    (e near 1, M near 0) they can take many more, or fail outright. Showing
    only one panel would let the report imply whichever answer suited it.

    ``typical`` and ``hard`` are each an (e, M) pair.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
    for ax, (e, M), tag in zip(axes, (typical, hard), ("typical", "hard corner")):
        plot_residual_history(history_df, e, M, ax=ax)
        ax.set_title(f"{tag}:  $e = {e:g}$,  $M = {M:g}$", fontsize=10)
    axes[1].set_ylabel("")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. Measured order against the paper's claim
# ----------------------------------------------------------------------
def plot_measured_vs_claimed_order(order_df, ax=None):
    """Bar chart of the measured order beside the order the paper claims.

    One pair of bars per solver. The error bar shows the spread across the
    test functions, which matters: a claim reproduced closely on every
    function is far stronger evidence than the same average pulled out of a
    wide scatter, and a reader should be able to tell those apart at a
    glance.

    Where the measurement could not be trusted - fewer than
    ``MIN_USABLE_TERMS`` usable errors, so the estimator never saw the true
    rate - the bar is hatched and labelled rather than left out.
    """
    _require(order_df, ["solver", "measured_order", "claimed_order"],
             "plot_measured_vs_claimed_order")

    created = ax is None
    if created:
        _, ax = plt.subplots()

    rows = []
    for solver, group in order_df.groupby("solver", sort=True):
        measured = pd.to_numeric(group["measured_order"], errors="coerce").dropna()
        claimed = pd.to_numeric(group["claimed_order"], errors="coerce").dropna()
        if "n_usable" in group.columns:
            usable = pd.to_numeric(group["n_usable"], errors="coerce").fillna(0)
            reliable = bool((usable >= MIN_USABLE_TERMS).all()) and not measured.empty
        else:
            reliable = not measured.empty
        rows.append({
            "solver": solver,
            "measured": measured.mean() if not measured.empty else math.nan,
            "spread": measured.std(ddof=0) if len(measured) > 1 else 0.0,
            "claimed": claimed.mean() if not claimed.empty else math.nan,
            "reliable": reliable,
            "n": len(measured),
        })

    # Sort by claimed order rather than by name. "newton, nwm9, nwm11" is
    # the progression the report argues about, and it matches the method
    # table. Sorting by name would put nwm11 before nwm9, which reads oddly.
    summary = (pd.DataFrame(rows)
               .sort_values("claimed", na_position="last")
               .reset_index(drop=True))
    positions = np.arange(len(summary))
    width = 0.38

    for offset, column, label, alpha in (
        (-width / 2, "measured", "measured", 1.0),
        (+width / 2, "claimed", "claimed by paper", 0.35),
    ):
        for i, row in summary.iterrows():
            value = row[column]
            if not np.isfinite(value):
                continue
            hatch = "//" if (column == "measured" and not row["reliable"]) else None
            ax.bar(
                positions[i] + offset, value, width,
                color=_solver_color(row["solver"]), alpha=alpha,
                hatch=hatch,
                edgecolor="white" if hatch is None else "0.25",
                linewidth=0.8,
                yerr=row["spread"] if column == "measured" else None,
                capsize=3 if column == "measured" else 0,
                error_kw={"ecolor": "0.25", "elinewidth": 1.0},
                label=label if i == 0 else None,
            )

    for i, row in summary.iterrows():
        if not np.isfinite(row["measured"]):
            ax.annotate("not\nmeasurable", (positions[i] - width / 2, 0.4),
                        ha="center", va="bottom", fontsize=7.5, color="0.3")
            continue
        if not row["reliable"]:
            ax.annotate("unreliable", (positions[i] - width / 2,
                                       row["measured"] + row["spread"]),
                        ha="center", va="bottom", fontsize=7.5, color="0.3")
        # At this scale a gap of 0.02 between the two bars is invisible - and
        # that invisibility IS the result. So print both numbers, or a reader
        # cannot tell "agrees to 3 digits" from "roughly the same".
        ax.annotate(f"{row['measured']:.3f}",
                    (positions[i] - width / 2, row["measured"] + row["spread"]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    xytext=(0, 9 if not row["reliable"] else 2),
                    textcoords="offset points")
        if np.isfinite(row["claimed"]):
            ax.annotate(f"{row['claimed']:.4f}",
                        (positions[i] + width / 2, row["claimed"]),
                        ha="center", va="bottom", fontsize=8, color="0.35",
                        xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(positions)
    ax.set_xticklabels(summary["solver"])
    ax.set_ylabel("convergence order $p$")
    ax.set_xlabel("")
    # Leave room at the top, or the label on the tallest bar gets cut off.
    tallest = np.nanmax(
        np.concatenate([summary["measured"].to_numpy(dtype=float),
                        summary["claimed"].to_numpy(dtype=float)])
    )
    if np.isfinite(tallest):
        ax.set_ylim(0, tallest * 1.15)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Measured convergence order vs. the source paper's claim",
                 fontsize=10)
    return ax.figure if created else ax


# ----------------------------------------------------------------------
# 3. Order across the eccentricity range
# ----------------------------------------------------------------------
def plot_order_vs_eccentricity(order_df, ax=None, log_1me=False):
    """Measured order plotted against eccentricity.

    This answers one of the project's main questions. An order is proved in
    the limit, for a well-behaved function. Kepler's equation gets much
    harder as e approaches 1, because f'(E) = 1 - e cos E falls towards zero
    near M = 0. Do the fast methods keep their claimed order across the whole
    range, or quietly fall apart in the corner? That is what this shows.

    Each solver's claimed order is drawn as a faint horizontal line, so the
    gap is visible without looking up a table. Points the estimator could
    not resolve are drawn hollow.

    Set ``log_1me=True`` to spread out the crowded high-eccentricity end on
    a log axis in (1 - e), which is where the interesting behaviour is.
    """
    _require(order_df, ["solver", "e", "measured_order"],
             "plot_order_vs_eccentricity")

    created = ax is None
    if created:
        _, ax = plt.subplots()

    for solver, group in order_df.groupby("solver", sort=True):
        group = group.sort_values("e")
        e = group["e"].to_numpy(dtype=float)
        order = pd.to_numeric(group["measured_order"], errors="coerce").to_numpy()
        x = (1.0 - e) if log_1me else e

        if "n_usable" in group.columns:
            usable = pd.to_numeric(group["n_usable"], errors="coerce").fillna(0)
            # .to_numpy() can return a read-only view of the table.
            reliable = np.array((usable >= MIN_USABLE_TERMS).to_numpy(), dtype=bool)
        else:
            reliable = np.isfinite(order)
        reliable = reliable & np.isfinite(order)

        color = _solver_color(solver)
        # Join up only the points the estimator could actually resolve. A
        # line through unreliable points claims a trend the data does not
        # support. Hollow markers say "measured here, but do not trust it".
        line_x = np.where(reliable, x, np.nan)
        ax.plot(line_x, np.where(reliable, order, np.nan), color=color,
                linewidth=1.5, label=str(solver), alpha=0.9)
        ax.plot(x[reliable], order[reliable], color=color, linestyle="none",
                marker="o", markersize=4)
        ax.plot(x[~reliable], order[~reliable], color=color, linestyle="none",
                marker="o", markersize=5, markerfacecolor="none")

        if "claimed_order" in group.columns:
            claimed = pd.to_numeric(group["claimed_order"],
                                    errors="coerce").dropna()
            if not claimed.empty:
                ax.axhline(claimed.iloc[0], color=color, linestyle=":",
                           linewidth=1.0, alpha=0.5)

    if log_1me:
        ax.set_xscale("log")
        ax.invert_xaxis()          # e increases to the right
        ax.set_xlabel(r"$1 - e$   (eccentricity increasing $\rightarrow$)")
    else:
        ax.set_xlabel("eccentricity $e$")

    # One bad estimate can be ten times too large and would squash every
    # real curve flat against the axis. Scale to the claimed orders and the
    # measurements worth believing, and let any outliers run off the top.
    reference = pd.to_numeric(order_df.get("claimed_order"), errors="coerce")
    believable = pd.to_numeric(order_df["measured_order"], errors="coerce")
    if "n_usable" in order_df.columns:
        mask = pd.to_numeric(order_df["n_usable"], errors="coerce").fillna(0)
        believable = believable[mask >= MIN_USABLE_TERMS]
    ceiling = max(
        float(reference.max()) if reference is not None and reference.notna().any() else 0.0,
        float(believable.max()) if believable.notna().any() else 0.0,
    )
    if ceiling > 0:
        ax.set_ylim(0, ceiling * 1.25)

    ax.set_ylabel("measured convergence order $p$")
    ax.set_title("Does the claimed order survive high eccentricity?", fontsize=10)
    ax.legend(fontsize=8, title="dotted = claimed   hollow = unreliable",
              title_fontsize=8, loc="center left")
    return ax.figure if created else ax
