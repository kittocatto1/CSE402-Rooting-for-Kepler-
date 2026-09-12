"""Convergence figures. Owner: Suchi.

These draw, they never compute.  Every function takes a DataFrame that
``evaluation/`` or ``experiments/`` produced and turns it into a figure, so
any figure in the report can be traced back to a result file.

Expected columns
----------------
``history_df``  one row per (solve, iteration), as written by
                ``io.results_io.save_history``:
                solver, guess, e, M, iteration, residual  (+ error, step)

``order_df``    one row per (solver, measurement), as produced by
                ``experiments.verification.verify_order_on_test_functions``
                or by an order table over the (e, M) grid:
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

#: Below this many usable terms the order estimator had no asymptotic window
#: and its output is not a measurement.  Such bars/markers are drawn hatched
#: or hollow rather than omitted - a missing bar reads as "no data", which is
#: a different and wronger claim than "measured, but unreliable".
MIN_USABLE_TERMS = 3

#: Relative size of one double-precision ulp, used to place the noise floor.
DOUBLE_EPS = 2.220446049250313e-16


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


def _guess_style(name: str) -> str:
    return GUESS_STYLES.get(str(name), "-")


# ----------------------------------------------------------------------
# 1. Residual histories
# ----------------------------------------------------------------------
def plot_residual_history(history_df, e: float, M: float, ax=None, atol=1e-12):
    """log10|f(E_n)| vs iteration, one line per solver, at one (e, M).

    The horizontal band marks the double-precision noise floor: ``f(E)`` is
    evaluated as ``E - e sin E - M``, so its own rounding error is about
    ``eps * max(|E|, |M|)``.  Below that line a curve is measuring
    floating-point noise, not convergence, and the report must not read
    anything into how far it drops.

    Residuals of exactly zero cannot be drawn on a log axis; they are plotted
    at the floor with a hollow marker and counted in the legend, because
    "hit zero" is a real and different outcome from "stalled".
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
        # groupby on a one-element list yields a 1-tuple key, not a scalar.
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
    """Two residual histories side by side: a typical (e, M) and a hard one.

    The story genuinely differs between the two regimes - in the ordinary
    operating range the high-order methods reach the floor in two or three
    steps, while in the pathological corner (e -> 1, M -> 0) they can take
    many more, or fail outright.  Showing only one panel would let the report
    imply whichever conclusion the author picked.

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
    """Bar chart: measured empirical order next to the paper's claim.

    One pair of bars per solver.  The error bar on the measured bar is the
    spread across test points - a claimed order reproduced to within noise on
    every function is a much stronger result than one averaged out of a wide
    scatter, and the report should be able to tell them apart at a glance.

    Where the measurement was unreliable - fewer than
    ``MIN_USABLE_TERMS`` usable errors before the precision floor, so the
    estimator never saw the asymptotic regime - the bar is drawn hatched and
    annotated rather than omitted.
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

    # Order by claimed order, not alphabetically: "newton, nwm9, nwm11" is
    # the progression the report argues about, and it matches the proposal's
    # method table. Alphabetical puts nwm11 before nwm9 and reads as noise.
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
        # At this scale a 0.02 gap between measured and claimed is invisible,
        # and "invisible" is the whole result - so print both numbers. A
        # reader must be able to tell "agrees to 3 digits" from "close-ish".
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
    # Headroom for the value labels, which otherwise clip on the tallest bar.
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
    """Measured order as a function of e.

    One of the project's central questions: a method's order is proved in the
    limit for a well-behaved f, but Kepler's equation stiffens as e -> 1
    (f'(E) = 1 - e cos E collapses towards zero near M = 0).  Whether the
    high-order methods hold their claimed order across the whole operating
    range, or quietly decay towards the pathological corner, is exactly what
    this figure is for.

    Each solver's claimed order is drawn as a faint horizontal reference so
    the gap is readable without cross-checking a table.  Points the estimator
    could not resolve are drawn hollow.

    Set ``log_1me=True`` to spread the crowded high-eccentricity end out on a
    log axis in (1 - e), which is where the interesting behaviour lives.
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
            # .to_numpy() can hand back a read-only view of the frame.
            reliable = np.array((usable >= MIN_USABLE_TERMS).to_numpy(), dtype=bool)
        else:
            reliable = np.isfinite(order)
        reliable = reliable & np.isfinite(order)

        color = _solver_color(solver)
        # Connect only the points the estimator could actually resolve. A line
        # drawn through unreliable points asserts a trend the data does not
        # support; hollow markers say "measured here, do not trust it".
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

    # A single unreliable estimate can be an order of magnitude out and would
    # flatten every real curve into the axis. Scale to the claims plus the
    # measurements worth believing, and let outliers run off the top.
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
