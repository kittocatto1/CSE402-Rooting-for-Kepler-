"""Tests for the report summary figures. Owner: Mahdi.

Kept in its own file, like the other plotting tests, so the owners of the
plotting modules do not collide in one shared file.

These check that every figure builds from a small synthetic sweep and that
the grid-set split recovers the three sets it is built from. They do not
check pixels; the figures are inspected by eye when the report is rebuilt.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from keplerbench.experiments import grid as grid_defs  # noqa: E402
from keplerbench.plotting import summary_plots as sp  # noqa: E402

#: A small grid: a 4 x 5 uniform block, a 3 x 3 corner and 6 RadVel points.
SPEC = {
    "type": "combined",
    "uniform": {"n_e": 4, "n_M": 5, "e_max": 0.99},
    "pathological": {"n_e": 3, "n_M": 3, "e_min": 0.9, "M_max": 0.1},
    "radvel": {"n_samples": 6, "seed": 0},
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sweep() -> pd.DataFrame:
    """Every solver and guess over the small grid, with a few failures."""
    rng = np.random.default_rng(1)
    rows = []
    for e, M in grid_defs.build_grid(SPEC):
        for solver in sp.ITERATIVE_SOLVERS:
            for guess in sp.GUESSES:
                failed = solver == "nwm11" and e > 0.99 and M < 1e-3
                rows.append({"solver": solver, "guess": guess, "e": e, "M": M,
                             "converged": not failed,
                             "iterations": 50 if failed else int(rng.integers(1, 7))})
        rows.append({"solver": "markley", "guess": "n/a", "e": e, "M": M,
                     "converged": True, "iterations": 0})
    return pd.DataFrame(rows)


@pytest.fixture
def safeguard_summary() -> pd.DataFrame:
    """The shape of results/safeguard/summary.csv, cut to the used columns."""
    rows = [("danby", 2), ("nwm9", 65), ("nwm9_memoryless", 13),
            ("nwm9_guarded", 13), ("nwm11", 74), ("nwm11_memoryless", 9),
            ("nwm11_guarded", 3)]
    df = pd.DataFrame(rows, columns=["solver", "n_failed_corner"])
    df["fail_pct_corner"] = 100.0 * df["n_failed_corner"] / 1524
    return df


def test_grid_set_recovers_the_three_blocks(sweep):
    points = sweep.drop_duplicates(["e", "M"])
    counts = sp.grid_set(points, SPEC).value_counts()
    assert counts.to_dict() == {"uniform": 20, "pathological": 9, "radvel": 6}


@pytest.mark.parametrize("build", [
    sp.plot_iterations_best_guess,
    sp.plot_iterations_all_combinations,
    sp.plot_iterations_by_set,
])
def test_set_aware_figures_build(sweep, build):
    assert isinstance(build(sweep, grid_spec=SPEC), plt.Figure)


@pytest.mark.parametrize("build", [
    sp.plot_failure_maps_by_solver,
    sp.plot_iterations_by_guess,
])
def test_whole_grid_figures_build(sweep, build):
    assert isinstance(build(sweep), plt.Figure)


def test_safeguard_figure_builds_and_labels_the_total(safeguard_summary):
    fig = sp.plot_safeguard_failures(safeguard_summary)
    assert isinstance(fig, plt.Figure)
    assert "(of 1,524)" in fig.axes[0].get_ylabel()


def test_safeguard_figure_rejects_a_missing_variant(safeguard_summary):
    without_guarded = safeguard_summary[safeguard_summary["solver"] != "nwm9_guarded"]
    with pytest.raises(ValueError, match="nwm9_guarded"):
        sp.plot_safeguard_failures(without_guarded)
