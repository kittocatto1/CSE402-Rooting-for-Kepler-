"""Tests for result-table writing and reading. Owner: Anisa."""

from __future__ import annotations

import json

import pytest

from keplerbench.core.types import IterationRecord, SolveResult
from keplerbench.io import results_io
from keplerbench.io.config import load_config


@pytest.fixture(autouse=True)
def isolated_results(tmp_path, monkeypatch):
    """Point results/ at a tmp dir so a test never touches a real run."""
    monkeypatch.setattr(results_io, "REPO_ROOT", tmp_path)
    return tmp_path


def _result(solver="danby", guess="simple", e=0.3, M=1.0, n_history=0, **over):
    fields = dict(solver=solver, guess=guess, e=e, M=M, E=1.2, converged=True,
                  iterations=3, residual=1e-16, error=2e-16, wall_time=1e-5,
                  failure=None, cost={"eval_points": 3, "sincos_pairs": 3})
    fields.update(over)
    fields["history"] = [
        IterationRecord(iteration=i, E=1.0 + i, residual=10.0 ** -i,
                        error=10.0 ** -i, step=None if i == 0 else 0.1,
                        cost={"sincos_pairs": i})
        for i in range(n_history)
    ]
    return SolveResult(**fields)


def test_results_round_trip():
    results_io.save_results([_result(), _result(solver="newton")], "exp")
    df = results_io.load_results("exp")
    assert len(df) == 2
    assert list(df["solver"]) == ["danby", "newton"]
    assert df["e"].dtype.kind == "f"


def test_the_n_a_guess_label_survives_the_round_trip():
    """"n/a" is on pandas' default missing-value list. Read back as NaN it
    would be dropped by every groupby, silently removing the closed-form
    solvers from every summary table in the report."""
    results_io.save_results([_result(solver="markley", guess="n/a")], "exp")
    df = results_io.load_results("exp")
    assert list(df["guess"]) == ["n/a"]
    assert df["guess"].notna().all()


def test_a_genuinely_empty_cell_is_still_nan():
    """The fix above must not turn every blank into the string ""."""
    results_io.save_results([_result(error=None, failure=None)], "exp")
    df = results_io.load_results("exp")
    assert df["error"].isna().all()
    assert df["failure"].isna().all()


def test_an_empty_run_still_writes_a_readable_table():
    """grid_benchmark's preflight probes save_results with no rows."""
    results_io.save_results([], "exp", "probe.csv")
    df = results_io.load_results("exp", "probe.csv")
    assert df.empty
    assert list(df.columns) == list(results_io.RESULT_COLUMNS)


def test_column_order_is_stable_across_solvers_with_different_counters():
    a = _result(solver="danby", cost={"sincos_pairs": 3})
    b = _result(solver="markley", cost={"sin_only": 1})
    results_io.save_results([a, b], "exp")
    columns = list(results_io.load_results("exp").columns)
    assert columns[:len(results_io.RESULT_COLUMNS)] == list(results_io.RESULT_COLUMNS)
    assert columns[len(results_io.RESULT_COLUMNS):] == ["cost_sin_only", "cost_sincos_pairs"]


def test_history_keeps_the_key_columns_on_every_row():
    """The convergence-order code groups a solve back out of the flat table."""
    results_io.save_history([_result(n_history=3), _result(solver="newton", n_history=2)],
                            "exp")
    df = results_io.load_results("exp", "history.csv")
    assert len(df) == 5
    assert list(df.columns)[:len(results_io.HISTORY_COLUMNS)] == \
        list(results_io.HISTORY_COLUMNS)
    one = df[df["solver"] == "danby"]
    assert list(one["iteration"]) == [0, 1, 2]
    assert one[["solver", "guess", "e", "M"]].nunique().tolist() == [1, 1, 1, 1]


def test_meta_records_provenance():
    cfg = load_config("configs/grid_benchmark.yaml")
    results_io.save_results([_result()], "exp", config=cfg)
    meta = results_io.load_meta("exp")
    assert meta["experiment"] == "exp"
    assert meta["n_rows"] == 1
    assert meta["config_name"] == "grid_benchmark"
    assert len(meta["config_fingerprint"]) == 8
    assert meta["written_at"]


def test_meta_is_per_table_not_per_directory():
    """raw.csv and history.csv are not always written in the same moment."""
    results_io.save_results([_result()], "exp")
    results_io.save_history([_result(n_history=2)], "exp")
    d = results_io.results_path("exp", "raw.csv").parent
    names = sorted(p.name for p in d.glob("*.json"))
    assert names == ["history.meta.json", "raw.meta.json"]
    assert json.loads((d / "raw.meta.json").read_text())["table"] == "raw.csv"


def test_meta_without_a_config_says_so_rather_than_guessing():
    results_io.save_results([_result()], "exp")
    meta = results_io.load_meta("exp")
    assert meta["config_fingerprint"] is None
    assert meta["config_name"] is None


def test_meta_path_names_exactly_what_save_results_wrote():
    """A caller that cleans up a table it wrote must be able to find the
    sidecar. grid_benchmark's preflight probe leaves a stray directory
    behind if it cannot - its rmdir only succeeds on an empty directory."""
    path = results_io.save_results([_result()], "exp", "probe.csv")
    sidecar = results_io.meta_path(path)
    assert sidecar.exists()

    path.unlink()
    sidecar.unlink()
    path.parent.rmdir()          # must now be empty, or this raises
