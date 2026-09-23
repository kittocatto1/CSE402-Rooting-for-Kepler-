"""Tests for the (e, M) grid figures. Owner: Mahdi.

Kept in its own file for the same reason as tests/test_plotting_convergence.py:
four of us own plotting modules and a shared file would collide.

These check the CONTRACT, not the pixels: that a figure is produced, that the
distinctions the report relies on survive into the drawing (failed is not
slow, unsampled is not failed), and that bad input fails loudly instead of
silently drawing nothing.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from keplerbench.plotting import grid_plots as gp  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def grid_df() -> pd.DataFrame:
    """A small sweep: 2 solvers x 2 guesses over a 2x2 (e, M) block.

    ``newton`` converges everywhere; ``nwm11`` fails at the corner point
    (e=0.99, M=0.01), which is the case every one of these tests turns on.
    """
    rows = []
    for e in (0.3, 0.99):
        for M in (0.01, 2.0):
            for solver in ("newton", "nwm11"):
                for guess in ("simple", "canonical"):
                    failed = (solver == "nwm11" and e == 0.99 and M == 0.01)
                    rows.append({
                        "solver": solver,
                        "guess": guess,
                        "e": e,
                        "M": M,
                        "converged": not failed,
                        "iterations": 50 if failed else (
                            4 if guess == "simple" else 3),
                        "residual": 1e-3 if failed else 1e-15,
                        "failure": None,
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def markley_df(grid_df) -> pd.DataFrame:
    """The sweep plus a closed-form solver run under the "n/a" pseudo-guess."""
    extra = [{"solver": "markley", "guess": "n/a", "e": e, "M": M,
              "converged": True, "iterations": 0, "residual": 1e-16,
              "failure": None}
             for e in (0.3, 0.99) for M in (0.01, 2.0)]
    return pd.concat([grid_df, pd.DataFrame(extra)], ignore_index=True)


# ----------------------------------------------------------------------
# plot_iterations_heatmap
# ----------------------------------------------------------------------
def test_iterations_heatmap_draws_a_mesh(grid_df):
    fig = gp.plot_iterations_heatmap(grid_df, solver="newton", guess="simple")
    ax = fig.axes[0]
    assert ax.collections, "no pcolormesh was drawn"
    assert "newton" in ax.get_title() and "simple" in ax.get_title()


def test_iterations_heatmap_overlays_failures_separately(grid_df):
    """A failed cell must be its own layer, not the top of the colormap.

    If it were merely the largest value in the mesh, the report could not
    tell "did not converge" from "converged after many iterations".
    """
    converging = gp.plot_iterations_heatmap(grid_df, "newton", "simple")
    failing = gp.plot_iterations_heatmap(grid_df, "nwm11", "simple")
    assert len(failing.axes[0].collections) > len(converging.axes[0].collections)


def test_iterations_heatmap_respects_a_shared_scale(grid_df):
    vmax = gp.shared_iteration_scale(grid_df)
    fig = gp.plot_iterations_heatmap(grid_df, "newton", "simple", vmax=vmax)
    assert fig.axes[0].collections[0].get_clim()[1] == pytest.approx(vmax)


def test_shared_iteration_scale_ignores_non_converged_rows(grid_df):
    """The 50-iteration failure must not set the top of the colour scale."""
    assert gp.shared_iteration_scale(grid_df) < 50.0


def test_iterations_heatmap_rejects_an_unrun_combination(grid_df):
    with pytest.raises(ValueError, match="no rows"):
        gp.plot_iterations_heatmap(grid_df, solver="newton", guess="napier")


def test_iterations_heatmap_rejects_a_missing_column(grid_df):
    with pytest.raises(KeyError, match="iterations"):
        gp.plot_iterations_heatmap(grid_df.drop(columns=["iterations"]),
                                   "newton", "simple")


# ----------------------------------------------------------------------
# plot_failure_map
# ----------------------------------------------------------------------
def test_failure_map_uses_log_axes_in_both_directions(grid_df):
    ax = gp.plot_failure_map(grid_df, solver="nwm11").axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_failure_map_separates_converged_from_failed(grid_df):
    ax = gp.plot_failure_map(grid_df, solver="nwm11").axes[0]
    labels = [c.get_label() for c in ax.collections]
    assert any("converged" in str(l) for l in labels)
    assert any("failed (1)" in str(l) for l in labels)


def test_failure_map_of_a_perfect_solver_reports_zero_failures(grid_df):
    ax = gp.plot_failure_map(grid_df, solver="newton").axes[0]
    assert any("failed (0)" in str(c.get_label()) for c in ax.collections)


def test_failure_map_marks_points_it_cannot_draw_on_log_axes():
    """e = 0 and M = 0 have no place on a log axis; say so, do not drop them
    silently - a quietly shrinking point count is how a real gap in coverage
    gets missed."""
    df = pd.DataFrame([
        {"solver": "newton", "guess": "simple", "e": 0.0, "M": 0.0,
         "converged": True, "iterations": 3},
        {"solver": "newton", "guess": "simple", "e": 0.5, "M": 1.0,
         "converged": True, "iterations": 3},
    ])
    ax = gp.plot_failure_map(df, solver="newton").axes[0]
    assert "not drawable" in ax.get_title()


def test_failure_map_can_be_restricted_to_one_guess(grid_df):
    fig = gp.plot_failure_map(grid_df, solver="nwm11", guess="simple")
    assert "simple" in fig.axes[0].get_title()


# ----------------------------------------------------------------------
# plot_guess_effect
# ----------------------------------------------------------------------
def test_guess_effect_draws_one_group_per_solver(grid_df):
    ax = gp.plot_guess_effect(grid_df).axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["newton", "nwm11"]


def test_guess_effect_excludes_closed_form_solvers(markley_df):
    """Markley ignores E0 and is run once under "n/a"; it has no guess effect
    and must not appear as an empty group in the figure."""
    ax = gp.plot_guess_effect(markley_df).axes[0]
    assert "markley" not in [t.get_text() for t in ax.get_xticklabels()]


def test_guess_effect_pairs_only_points_both_guesses_solved(grid_df):
    """nwm11 fails at the corner under BOTH guesses, so that point is simply
    absent; the remaining three points each save one iteration."""
    ax = gp.plot_guess_effect(grid_df).axes[0]
    heights = [bar.get_height() for bar in ax.patches]
    assert heights == pytest.approx([1.0, 1.0])


def test_guess_effect_ignores_a_point_only_one_guess_solved():
    """A guess must not look good because it failed on the hard points and so
    never paid their iteration cost."""
    rows = [
        # both solved: canonical saves 2 iterations
        {"solver": "newton", "guess": "simple", "e": 0.3, "M": 1.0,
         "converged": True, "iterations": 5},
        {"solver": "newton", "guess": "canonical", "e": 0.3, "M": 1.0,
         "converged": True, "iterations": 3},
        # only the baseline solved: must not enter the average at all
        {"solver": "newton", "guess": "simple", "e": 0.99, "M": 0.01,
         "converged": True, "iterations": 40},
        {"solver": "newton", "guess": "canonical", "e": 0.99, "M": 0.01,
         "converged": False, "iterations": 50},
    ]
    ax = gp.plot_guess_effect(pd.DataFrame(rows)).axes[0]
    assert [bar.get_height() for bar in ax.patches] == pytest.approx([2.0])


def test_guess_effect_legend_names_the_guesses_not_the_solvers(grid_df):
    """Colour encodes the solver in this figure, so a legend built from the
    bars would show every guess in whichever solver sorts first. The key must
    list the guesses and lean on hatch alone."""
    ax = gp.plot_guess_effect(grid_df).axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["canonical"]


def test_guess_effect_rejects_an_unknown_baseline(grid_df):
    with pytest.raises(ValueError, match="baseline"):
        gp.plot_guess_effect(grid_df, baseline="napier")


def test_guess_effect_rejects_an_all_closed_form_frame():
    df = pd.DataFrame([
        {"solver": "markley", "guess": "n/a", "e": 0.3, "M": 1.0,
         "converged": True, "iterations": 0},
    ])
    with pytest.raises(ValueError, match="guess-independent"):
        gp.plot_guess_effect(df)


# ----------------------------------------------------------------------
# Reading raw.csv back from disk
# ----------------------------------------------------------------------
def test_n_a_label_survives_a_csv_round_trip(markley_df, tmp_path):
    """"n/a" is on pd.read_csv's default missing-value list, so a raw.csv
    read back the ordinary way has NaN for markley's guess. The figures must
    still treat it as the guess-independent label, not crash on it."""
    path = tmp_path / "raw.csv"
    markley_df.to_csv(path, index=False)
    df = pd.read_csv(path)
    assert df.loc[df["solver"] == "markley", "guess"].isna().all()

    ax = gp.plot_guess_effect(df).axes[0]
    assert "markley" not in [t.get_text() for t in ax.get_xticklabels()]
    gp.plot_iterations_heatmap(df, "markley", gp.GUESS_INDEPENDENT_LABEL)
    gp.plot_failure_map(df, "markley", guess=gp.GUESS_INDEPENDENT_LABEL)


def test_a_missing_guess_on_an_iterative_solver_fails_loudly(grid_df):
    """Only closed-form solvers may lack a guess; anywhere else a NaN is a
    real gap, and filing it under "n/a" would drop it from the comparison."""
    df = grid_df.copy()
    df.loc[df.index[0], "guess"] = None
    with pytest.raises(ValueError, match="no guess label"):
        gp.plot_guess_effect(df)
