"""Tests for the cost-accounting figures. Owner: Dipit.

Kept in its own file, like the other plotting tests, so owners do not
collide. These check the contract, not the pixels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from keplerbench.evaluation.cost_model import per_iteration_cost_table  # noqa: E402
from keplerbench.experiments.runner import solve_one  # noqa: E402
from keplerbench.core.registry import get_guess, get_solver  # noqa: E402
from keplerbench.plotting import cost_plots as cp  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def cost_table() -> pd.DataFrame:
    return pd.DataFrame(per_iteration_cost_table())


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """A real little sweep, with the error column filled from a tight Newton
    solve (good to ~1e-16, plenty for digit counts in a test)."""
    rows = []
    for e in (0.2, 0.7, 0.95):
        for M in (0.05, 1.0, 2.5):
            ref = solve_one(get_solver("newton"), get_guess("simple"), e, M,
                            tol=0.0, max_iter=60).E
            for solver, guess in (("newton", "simple"), ("danby", "napier"),
                                  ("nwm9", "canonical"), ("markley", "simple")):
                r = solve_one(get_solver(solver), get_guess(guess), e, M,
                              E_reference=ref)
                r.guess = "n/a" if solver == "markley" else guess
                rows.append(r.to_row())
    df = pd.DataFrame(rows)
    df.loc[df["solver"] == "markley", "guess"] = float("nan")  # as read_csv does
    return df


def test_cost_breakdown_draws_both_costings(cost_table):
    fig = cp.plot_cost_breakdown(cost_table)
    assert len(fig.axes) == 2
    # one bar segment per (method, kind) in each panel
    assert all(len(ax.patches) == 4 * len(cost_table) for ax in fig.axes)


def test_cost_breakdown_rejects_missing_columns(cost_table):
    with pytest.raises(KeyError):
        cp.plot_cost_breakdown(cost_table.drop(columns=["sin only"]))


def test_cost_vs_accuracy(raw_df, cost_table):
    fig = cp.plot_cost_vs_accuracy(raw_df, cost_table)
    assert len(fig.axes) == 2
    # markley has no claimed order, so it is named as left out, not dropped
    assert "markley" in fig.axes[1].get_title()


def test_cost_vs_accuracy_needs_reference_errors(raw_df):
    with pytest.raises(ValueError):
        cp.plot_cost_vs_accuracy(raw_df.assign(error=float("nan")))


def test_wallclock_vs_cost_joins_timing_to_raw(raw_df):
    timing = raw_df[["solver", "guess", "e", "M"]].copy()
    timing["seconds"] = [1e-6 * (i + 1) for i in range(len(timing))]
    fig = cp.plot_wallclock_vs_cost(timing, raw_df)
    assert "Pearson r" in fig.axes[0].get_title()


def test_wallclock_vs_cost_needs_counters(raw_df):
    timing = raw_df[["solver", "guess", "e", "M"]].assign(seconds=1e-6)
    with pytest.raises(KeyError):
        cp.plot_wallclock_vs_cost(timing)
