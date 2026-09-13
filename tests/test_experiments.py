"""Tests for the pipeline and the grid. Owner: Mahdi."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.grid import build_grid, pathological_grid, uniform_grid
from keplerbench.experiments.runner import run_sweep, solve_one


def test_solve_one_labels_the_guess():
    r = solve_one(get_solver("newton"), get_guess("simple"), e=0.2, M=1.0)
    assert r.guess == "simple"
    assert r.solver == "newton"


def test_guess_cost_is_not_charged_to_the_iteration():
    """The guess layer must be measured separately from the iteration, so
    the counters start at zero when the first step runs."""
    r = solve_one(get_solver("newton"), get_guess("simple"),
                  e=0.5, M=1.0, tol=0.0, max_iter=1, record_history=True)
    # iteration 0 record is the guess itself: one sin-only residual check.
    assert r.history[0].cost["sincos_pairs"] == 0


def test_uniform_grid_shape():
    with skip_if_unimplemented():
        pts = uniform_grid(n_e=5, n_M=4)
        assert len(pts) == 20
        assert all(0.0 <= e < 1.0 and 0.0 <= M <= math.pi for e, M in pts)


def test_pathological_grid_is_actually_in_the_corner():
    with skip_if_unimplemented():
        pts = pathological_grid()
        assert all(e >= 0.9 for e, _ in pts)
        assert all(M <= 0.1 for _, M in pts)


def test_build_grid_dispatch():
    with skip_if_unimplemented():
        pts = build_grid({"type": "uniform", "uniform": {"n_e": 3, "n_M": 3}})
        assert len(pts) == 9


def test_run_sweep_covers_every_combination():
    with skip_if_unimplemented():
        results = run_sweep(["newton"], ["simple"], [(0.1, 1.0), (0.5, 2.0)])
        assert len(results) == 2


# ----------------------------------------------------------------------
# Robustness metrics (evaluation/robustness.py). Owner: Mahdi.
# ----------------------------------------------------------------------
def _fake_results():
    import pandas as pd

    return pd.DataFrame([
        # ordinary region, fine
        dict(solver="newton", guess="simple", e=0.5, M=1.0,
             converged=True, failure=None, error=1e-15),
        # hard corner, ran out of iterations ("slow")
        dict(solver="newton", guess="simple", e=0.99, M=0.01,
             converged=False, failure=None, error=1e-2),
        # hard corner, blew up ("broken")
        dict(solver="newton", guess="simple", e=0.99, M=0.05,
             converged=False, failure="OverflowError", error=None),
        # hard corner, converged to the wrong root
        dict(solver="newton", guess="simple", e=0.95, M=0.02,
             converged=True, failure=None, error=1e-3),
    ])


def test_failure_rates_separates_slow_from_broken():
    from keplerbench.evaluation.robustness import failure_rates

    with skip_if_unimplemented():
        row = failure_rates(_fake_results()).iloc[0]
        assert row["n_points"] == 4
        assert row["n_converged"] == 2
        assert row["n_max_iter_hit"] == 1
        assert row["n_exception"] == 1
        assert row["failure_rate"] == pytest.approx(0.5)


def test_failure_rates_by_region_finds_the_hard_corner():
    from keplerbench.evaluation.robustness import failure_rates_by_region

    with skip_if_unimplemented():
        out = failure_rates_by_region(_fake_results()).set_index("region")
        assert out.loc["hard_corner", "n_points"] == 3
        assert out.loc["ordinary", "failure_rate"] == pytest.approx(0.0)


def test_wrong_root_rate_uses_the_error_not_the_residual():
    from keplerbench.evaluation.robustness import wrong_root_rate

    with skip_if_unimplemented():
        row = wrong_root_rate(_fake_results()).iloc[0]
        assert row["n_converged"] == 2
        assert row["n_wrong_root"] == 1
        assert row["wrong_root_rate"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# grid_benchmark.run - the main experiment.
#
# Its io layer (load_config, save_results, save_history) and its reference
# roots belong to Anisa and are still stubs, so these tests substitute test
# doubles for them. That checks the orchestration this module is actually
# responsible for - grid -> sweep -> history sub-grid -> timing -> summary -
# without waiting on another track, and without duplicating her work.
# ----------------------------------------------------------------------
class _FakeIO:
    """Records what grid_benchmark asked to have written."""

    def __init__(self):
        self.results: dict[str, list] = {}
        self.history: dict[str, list] = {}
        self.frames: dict[str, object] = {}


@pytest.fixture
def fake_io(monkeypatch, tmp_path):
    from keplerbench.experiments import grid_benchmark as gb

    io = _FakeIO()

    # Mirrors the real contract: each experiment gets its OWN directory under
    # results/. Flattening them into one directory here would let a tidy-up
    # in one experiment delete another's files - which is exactly what the
    # real layout prevents.
    def results_path(experiment, filename):
        d = tmp_path / experiment
        d.mkdir(parents=True, exist_ok=True)
        return d / filename

    def save_results(results, experiment, filename="raw.csv"):
        io.results[filename] = list(results)
        path = results_path(experiment, filename)
        path.write_text("")
        return path

    def save_history(results, experiment, filename="history.csv"):
        io.history[filename] = list(results)
        return results_path(experiment, filename)

    monkeypatch.setattr(gb, "save_results", save_results)
    monkeypatch.setattr(gb, "save_history", save_history)
    monkeypatch.setattr(gb, "results_path", results_path)
    return io


def _config(**overrides):
    from keplerbench.io.config import ExperimentConfig

    settings = dict(
        name="grid_benchmark",
        solvers=["newton", "danby"],
        guesses=["simple", "canonical"],
        grid={"type": "uniform", "uniform": {"n_e": 3, "n_M": 3}},
        tol=1e-12,
        max_iter=50,
        use_reference=False,
        timing_repeats=2,
        extra={},
    )
    settings.update(overrides)
    return ExperimentConfig(**settings)


def _use_config(monkeypatch, cfg):
    from keplerbench.experiments import grid_benchmark as gb
    monkeypatch.setattr(gb, "load_config", lambda path: cfg)


def test_grid_benchmark_writes_one_row_per_combination(monkeypatch, fake_io):
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config())
    gb.run("ignored.yaml")

    rows = fake_io.results["raw.csv"]
    assert len(rows) == 9 * 2 * 2          # points x solvers x guesses
    assert {r.solver for r in rows} == {"newton", "danby"}


def test_grid_benchmark_limit_truncates_the_grid(monkeypatch, fake_io):
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config())
    gb.run("ignored.yaml", limit=2)

    assert len({(r.e, r.M) for r in fake_io.results["raw.csv"]}) == 2


def test_grid_benchmark_keeps_history_off_for_the_main_sweep(monkeypatch,
                                                             fake_io):
    """Recording history over the full grid would cost gigabytes."""
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config())
    gb.run("ignored.yaml")

    assert all(r.history == [] for r in fake_io.results["raw.csv"])


def test_grid_benchmark_runs_the_history_subgrid_separately(monkeypatch,
                                                            fake_io):
    """The convergence figures need per-iteration records, so the sub-grid is
    run a second time with history ON."""
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config(
        extra={"history_subgrid": {"e": [0.3, 0.9], "M": [0.1, 1.0]}}))
    gb.run("ignored.yaml")

    recorded = fake_io.history["history.csv"]
    assert len({(r.e, r.M) for r in recorded}) == 4
    assert any(r.history for r in recorded)


def test_grid_benchmark_times_only_a_subset(monkeypatch, fake_io, tmp_path):
    """Timing every grid point is far too slow and is not needed."""
    import pandas as pd
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config(extra={"timing_points": 2}))
    gb.run("ignored.yaml")

    timing = pd.read_csv(tmp_path / "grid_benchmark" / "timing.csv")
    assert len({(e, M) for e, M in zip(timing.e, timing.M)}) == 2
    assert (timing["seconds"] > 0).all()


def test_grid_benchmark_survives_a_stubbed_aggregate(monkeypatch, fake_io,
                                                     tmp_path):
    """raw.csv is the deliverable and is already written by then; a stub in
    someone else's aggregate module must not throw the sweep away."""
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config())
    with pytest.warns(UserWarning, match="summarise_grid"):
        gb.run("ignored.yaml")

    assert fake_io.results["raw.csv"]
    assert (tmp_path / "grid_benchmark" / "summary.csv").exists()


def test_grid_benchmark_preflight_fails_before_the_sweep(monkeypatch):
    """A stubbed save_results must be caught up front, not after an hour of
    sweeping."""
    from keplerbench.experiments import grid_benchmark as gb

    _use_config(monkeypatch, _config())
    swept = []
    monkeypatch.setattr(gb, "run_sweep",
                        lambda *a, **k: swept.append(1) or [])

    with pytest.raises(NotImplementedError, match="save_results"):
        gb.run("ignored.yaml")
    assert not swept, "preflight ran the sweep before checking dependencies"


def test_grid_benchmark_reports_a_stubbed_load_config_as_a_status(monkeypatch):
    """Running the script today must say who it is waiting on, not hand the
    runner a bare traceback from inside someone else's module."""
    from keplerbench.experiments import grid_benchmark as gb

    with pytest.raises(NotImplementedError, match="load_config") as excinfo:
        gb.run("configs/grid_benchmark.yaml")
    assert "still stubs" in str(excinfo.value)
