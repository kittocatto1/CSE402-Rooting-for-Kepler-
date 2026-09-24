"""Tests for Work Plan step 2 - the verification gate. Owner: Suchi.

The gate decides whether a solver is allowed into the grid benchmark, so a
gate that passes something broken is worse than no gate. These check the two
things that would make it useless: that it does not mistake divergence for a
wrong answer, and that it does not mistake a wrong answer for divergence.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.base import KeplerSolver
from keplerbench.core.types import SolveResult
from keplerbench.experiments import verification
from keplerbench.experiments.verification import (
    BRANCH_TOL,
    kepler_check_points,
    verify_kepler_correctness,
)


def test_check_points_cover_the_whole_range_including_the_corner():
    """A sample that misses e -> 1 would never see a branch jump."""
    points = kepler_check_points(400)
    assert len(points) > 100
    eccentricities = [e for e, _ in points]
    assert min(eccentricities) < 0.1
    assert max(eccentricities) > 0.999, "pathological corner not sampled"
    assert all(0.0 <= e < 1.0 for e in eccentricities)
    assert all(math.isfinite(M) for _, M in points)


def test_check_points_are_deterministic():
    """The RadVel block is random; a moving sample would make the summary
    table irreproducible from the config, which the house rules forbid."""
    assert kepler_check_points(400, seed=0) == kepler_check_points(400, seed=0)
    assert kepler_check_points(400, seed=0) != kepler_check_points(400, seed=1)


@pytest.mark.parametrize("solver_name",
                         ["newton", "danby", "markley", "nwm9", "nwm11"])
def test_every_solver_finds_the_right_root_where_it_converges(solver_name):
    """The check the gate exists for: no silent branch jumps.

    Non-convergence is allowed here - that is the robustness metric's
    business, and solvers genuinely do fail in the pathological corner. What
    is not allowed is converging to a DIFFERENT root, where |f(E)| is tiny,
    the convergence flag is True, and every downstream number is wrong.
    """
    with skip_if_unimplemented():
        report = verify_kepler_correctness(solver_name, n_points=200)
        assert report["n_wrong_root"] == 0, report["first_wrong_point"]
        assert report["passed"]
        assert report["max_abs_error"] < 1e-8, report
        assert report["n_converged"] > 0


class _DivergingSolver(KeplerSolver):
    """Runs out of iterations and returns a huge iterate, without raising."""

    name = "diverger"
    theoretical_order = None

    def solve(self, problem, E0, tol=1e-14, max_iter=50,
              record_history=False, E_reference=None):
        E = 8.8e12
        return SolveResult(
            solver=self.name, guess="", e=problem.e, M=problem.M, E=E,
            converged=False, iterations=max_iter, residual=abs(E),
            error=None if E_reference is None else abs(E - E_reference),
            failure=None,
        )


class _WrongRootSolver(KeplerSolver):
    """Converges cleanly onto the NEXT branch: small residual, wrong answer."""

    name = "wrongroot"
    theoretical_order = None

    def solve(self, problem, E0, tol=1e-14, max_iter=50,
              record_history=False, E_reference=None):
        true_root = E_reference if E_reference is not None else problem.M
        E = true_root + 2 * math.pi
        return SolveResult(
            solver=self.name, guess="", e=problem.e, M=problem.M, E=E,
            converged=True, iterations=3, residual=1e-16,
            error=None if E_reference is None else abs(E - E_reference),
            failure=None,
        )


@pytest.fixture
def _registered(monkeypatch):
    """Put the two fakes in the registry for the duration of one test."""
    import keplerbench.core.registry as registry

    originals = dict(registry._SOLVERS)
    registry._SOLVERS["diverger"] = _DivergingSolver
    registry._SOLVERS["wrongroot"] = _WrongRootSolver
    yield
    registry._SOLVERS.clear()
    registry._SOLVERS.update(originals)


def test_divergence_is_reported_as_failure_not_as_a_wrong_root(_registered):
    """The bug this test exists for was real.

    An earlier version accepted any run that did not raise, so a solver that
    merely exhausted max_iter and walked off to E = 8.8e12 was counted as
    having answered - and then flagged as a WRONG ROOT. That inverts the
    meaning of the gate: an honest failure reported as dishonesty.
    """
    report = verify_kepler_correctness("diverger", n_points=40)
    assert report["n_wrong_root"] == 0
    assert report["n_converged"] == 0
    assert report["n_failed"] == report["n_points"]
    assert report["passed"], "divergence alone must not fail the gate"


def test_a_converged_wrong_root_fails_the_gate(_registered):
    """The complementary direction: small residual must not buy a pass."""
    report = verify_kepler_correctness("wrongroot", n_points=40)
    assert report["n_wrong_root"] == report["n_points"]
    assert not report["passed"]
    assert report["first_wrong_point"] is not None
    # 2*pi away is far beyond the branch threshold.
    assert 2 * math.pi > BRANCH_TOL


def test_run_writes_both_tables_and_reports_pass(tmp_path, monkeypatch, capsys):
    """End to end: config in, CSVs out, a verdict per solver on stdout.

    Kept small - two solvers, few points, low precision - because this is
    testing the plumbing, not the numerics. The order and correctness values
    themselves are covered by their own tests.
    """
    config = tmp_path / "verification.yaml"
    config.write_text(
        "name: verification\n"
        "solvers: [newton]\n"
        "guesses: [simple]\n"
        "tol: 0.0\n"
        "max_iter: 8\n"
        "record_history: true\n"
        "use_reference: true\n"
        "extra:\n"
        "  working_precision_dps: 300\n"
        "  kepler_check_points: 40\n"
        "  kepler_check_tol: 1.0e-13\n"
    )

    written: dict[str, object] = {}

    def fake_results_path(experiment, filename):
        path = tmp_path / experiment / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        written[filename] = path
        return path

    monkeypatch.setattr(verification, "results_path", fake_results_path)
    verification.run(str(config))

    out = capsys.readouterr().out
    assert "PASS" in out and "newton" in out

    summary = pd.read_csv(written["summary.csv"])
    assert list(summary["solver"]) == ["newton"]
    assert bool(summary.loc[0, "passed"])
    assert summary.loc[0, "measured_order_mean"] == pytest.approx(2.0, abs=1e-6)
    assert summary.loc[0, "kepler_n_wrong_root"] == 0

    order = pd.read_csv(written["order.csv"])
    assert len(order) == 8, "one row per paper test function"
    assert set(order["solver"]) == {"newton"}


def test_run_skips_an_unimplemented_solver_instead_of_aborting(
        tmp_path, monkeypatch):
    """The team works in parallel; one stub must not block everyone's gate."""
    import keplerbench.core.registry as registry

    class Stub(KeplerSolver):
        name = "stubbed"
        theoretical_order = 3.0

        def solve(self, *args, **kwargs):
            raise NotImplementedError("Stub.solve: not written yet")

    originals = dict(registry._SOLVERS)
    registry._SOLVERS["stubbed"] = Stub
    config = tmp_path / "verification.yaml"
    config.write_text(
        "name: verification\n"
        "solvers: [stubbed]\n"
        "guesses: [simple]\n"
        "tol: 0.0\n"
        "max_iter: 6\n"
        "use_reference: true\n"
        "extra:\n"
        "  working_precision_dps: 200\n"
        "  kepler_check_points: 20\n"
    )
    monkeypatch.setattr(
        verification, "results_path",
        lambda experiment, filename: tmp_path / filename)
    try:
        with pytest.warns(UserWarning, match="skipping stubbed"):
            verification.run(str(config))
        summary = pd.read_csv(tmp_path / "summary.csv")
        assert summary.empty
    finally:
        registry._SOLVERS.clear()
        registry._SOLVERS.update(originals)
