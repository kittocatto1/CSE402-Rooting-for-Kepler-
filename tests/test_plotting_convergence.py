"""Tests for the convergence figures. Owner: Suchi.

Kept in its own file rather than a shared tests/test_plotting.py, because
four of us own plotting modules and a single shared file would collide the
way tests/test_evaluation.py already does.

These check the CONTRACT, not the pixels: that a figure is produced, and
that bad input fails loudly instead of silently drawing nothing. A blank
figure that reaches the report is worse than a crash here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from keplerbench.plotting import convergence_plots as cp  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def history_df() -> pd.DataFrame:
    rows = []
    for solver, residuals in (
        ("newton", [1e-1, 1e-3, 1e-7, 1e-15]),
        ("nwm11", [1e-1, 1e-11, 0.0]),          # exact zero is a real outcome
    ):
        for n, residual in enumerate(residuals):
            rows.append({"solver": solver, "guess": "simple", "e": 0.6, "M": 2.0,
                         "iteration": n, "E": 2.4, "residual": residual})
    return pd.DataFrame(rows)


@pytest.fixture
def order_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"solver": "newton", "function": "phi1", "measured_order": 2.0,
         "claimed_order": 2.0, "n_usable": 7},
        {"solver": "newton", "function": "phi2", "measured_order": 2.0,
         "claimed_order": 2.0, "n_usable": 7},
        {"solver": "nwm11", "function": "phi1", "measured_order": 10.77,
         "claimed_order": 10.7446, "n_usable": 4},
        {"solver": "nwm11", "function": "phi2", "measured_order": 10.72,
         "claimed_order": 10.7446, "n_usable": 4},
    ])


# ----------------------------------------------------------------------
def test_residual_history_draws_every_solver(history_df):
    ax = cp.plot_residual_history(history_df, e=0.6, M=2.0).axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert any("newton" in str(l) for l in labels)
    assert any("nwm11" in str(l) for l in labels)
    assert ax.get_yscale() == "log"


def test_residual_history_keeps_exact_zeros_visible(history_df):
    """A residual of exactly 0 cannot go on a log axis, but dropping it
    silently would turn "hit the root exactly" into "stopped early"."""
    ax = cp.plot_residual_history(history_df, e=0.6, M=2.0).axes[0]
    assert any("exact zero" in str(line.get_label()) for line in ax.get_lines())


def test_residual_history_fails_loudly_on_an_absent_point(history_df):
    with pytest.raises(ValueError, match="no rows for"):
        cp.plot_residual_history(history_df, e=0.123, M=4.56)


def test_measured_vs_claimed_orders_bars_by_claimed_order(order_df):
    ax = cp.plot_measured_vs_claimed_order(order_df).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["newton", "nwm11"]


def test_measured_vs_claimed_handles_an_unmeasurable_solver(order_df):
    """nan must render as an annotated gap, not crash and not vanish."""
    extra = pd.DataFrame([{"solver": "nwm9", "function": "phi1",
                           "measured_order": float("nan"),
                           "claimed_order": 8.8989, "n_usable": 1}])
    ax = cp.plot_measured_vs_claimed_order(
        pd.concat([order_df, extra], ignore_index=True)).axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("not" in t for t in texts), texts


def test_order_vs_eccentricity_only_connects_reliable_points():
    """A line through unreliable points asserts a trend the data lacks."""
    df = pd.DataFrame([
        {"solver": "nwm9", "e": 0.1, "measured_order": 8.9,
         "claimed_order": 8.8989, "n_usable": 5},
        {"solver": "nwm9", "e": 0.9, "measured_order": 8.9,
         "claimed_order": 8.8989, "n_usable": 5},
        {"solver": "nwm9", "e": 0.999, "measured_order": 40.0,
         "claimed_order": 8.8989, "n_usable": 1},      # noise, unreliable
    ])
    fig = cp.plot_order_vs_eccentricity(df)
    ax = fig.axes[0]
    # The outlier must not set the scale.
    assert ax.get_ylim()[1] < 20, ax.get_ylim()


@pytest.mark.parametrize("fn,df,kwargs", [
    (cp.plot_residual_history, pd.DataFrame({"solver": ["x"]}), {"e": 0, "M": 0}),
    (cp.plot_measured_vs_claimed_order, pd.DataFrame({"solver": ["x"]}), {}),
    (cp.plot_order_vs_eccentricity, pd.DataFrame({"solver": ["x"]}), {}),
])
def test_missing_columns_fail_loudly(fn, df, kwargs):
    with pytest.raises(KeyError, match="missing"):
        fn(df, **kwargs)
