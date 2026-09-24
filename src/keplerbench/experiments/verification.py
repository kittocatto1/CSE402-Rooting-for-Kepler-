"""Work Plan step 2 - verify each implementation before trusting it.

The rule for this project: a solver does not enter the grid benchmark until
its MEASURED convergence order matches what its source paper claims, on the
paper's OWN test functions.  This is what stops a transcription bug from
being reported as "the with-memory method underperforms".

Owner: Suchi.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Sequence

import mpmath as mp
import pandas as pd

from keplerbench.core.counters import CostCounter
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.evaluation.convergence_order import order_from_history_detail
from keplerbench.experiments.grid import (pathological_grid,
                                          radvel_operating_grid, uniform_grid)
from keplerbench.experiments.runner import solve_one
from keplerbench.io.config import load_config
from keplerbench.io.results_io import results_path
from keplerbench.reference.mpmath_reference import reference_root

__all__ = [
    "BRANCH_TOL",
    "kepler_check_points",
    "measure_order_on_kepler",
    "GenericProblem",
    "PAPER_TEST_FUNCTIONS",
    "VERIFICATION_DPS",
    "refine_root",
    "verify_order_on_test_functions",
    "verify_kepler_correctness",
    "run",
]

#: Working precision for the order measurement.  Mirrors
#: extra.working_precision_dps in configs/verification.yaml.
#:
#: This has to be far larger than feels reasonable, and the reason is worth
#: stating.  An order-p method squares-and-then-some each step: from an error
#: of 1e-1 an order-9 method passes 1e-9, 1e-81, 1e-729...  The estimator
#: needs three consecutive usable errors for ONE estimate, so the working
#: precision must outrun p**3 digits or the sequence hits the floor before a
#: single triple exists.
#:
#: Measured on NWM9 (claimed 8.8989) over the paper's eight test functions:
#:
#:     dps=100    measured 8.87-10.19, only 3 usable terms  -> FAILS
#:     dps=1000   measured 8.87-9.04                        -> 0.55 s
#:     dps=2000   measured 8.87-8.98                        -> 2.14 s
#:     dps=3000   identical to dps=2000                     -> 4.70 s
#:
#: 2000 is where the estimate stops improving.  NWM11 (order 10.7446) burns
#: through digits faster still, so do not lower this.
VERIFICATION_DPS = 2000

#: Test functions from the with-memory papers. Each entry is
#: (name, f, published_root, x0).
#:
#: These eight are the numerical examples of Mittal, Panday, Jantschi &
#: Bolundut, AIMS Mathematics 10(3), 5421-5443, 2025 (doi:10.3934/math.2025250),
#: Section 3 - the source for NWM10 and NWM11.  Same functions, same starting
#: points as the paper, otherwise comparing against their reported order is
#: meaningless.
#:
#: ``published_root`` is the paper's own 5-6 digit value.  It is NOT accurate
#: enough to measure an order-10 method against; :func:`refine_root` sharpens
#: it to the working precision and that refined value is what the estimator
#: uses.  The published value is kept so a transcription error shows up as a
#: refinement that walks away from it.
#:
#: NOTE: the functions are written with mpmath primitives so they evaluate at
#: whatever precision the active context is using.  NWM9 comes from a
#: different paper (Mathematics 12(22), 3490) with its own test set; add it
#: as a separate list rather than mixing the two.
PAPER_TEST_FUNCTIONS: list[tuple[str, Callable[[Any], Any], float, float]] = [
    ("phi1", lambda s: s**10 + 4 * s**5 - 15 * s**2 + 2, 1.24903, 1.2),
    ("phi2", lambda s: mp.cos(s) ** 2 - mp.sin(s) + s, -1.09775, -0.9),
    ("phi3", lambda s: mp.e ** (s - 5) - s**3 + mp.sin(s) - 1, -1.24861, -1.1),
    ("phi4", lambda s: mp.sin(s**3 + s**2 + 1) + s**2 - 5, 2.30388, 2.3),
    ("phi5", lambda s: mp.e ** (s**2) - s**3 + s**2 + s - 7, 1.35666, 1.3),
    ("phi6", lambda s: s**7 - 4 * s**4 + s - 1, 1.57494, 1.5),
    ("phi7", lambda s: mp.e ** (s**3 + mp.cos(s) + 1) - s**2 + s + 1, -1.07875, -1.2),
    ("phi8", lambda s: mp.e ** (s**2) + mp.sin(s) - mp.cos(s) - 1, 0.54177, 0.4),
]


# ----------------------------------------------------------------------
# Driving a Kepler solver with an arbitrary f
# ----------------------------------------------------------------------
class GenericProblem:
    """Adapter letting a solver run on any f, for verification only.

    The solvers are written against :class:`~keplerbench.core.types.KeplerProblem`,
    but the papers report their orders on generic nonlinear test functions.
    This exposes the same evaluation surface so ``IterativeSolver.solve`` can
    drive a solver unchanged.

    It lives here rather than in ``core/`` on purpose: the benchmark path
    stays Kepler-specific, and nothing on the measurement path can
    accidentally import a generic problem into the grid run.

    Cost accounting
    ---------------
    The counters are incremented so that solvers which report synthesised
    derivatives keep working, but on a generic function ``sincos_pairs`` and
    ``sin_only`` merely mean "paired evaluation" and "value-only evaluation".
    They carry no Kepler meaning here, and verification output must never
    feed the Section 4.1 cost tables.
    """

    def __init__(
        self,
        f: Callable[[Any], Any],
        fprime: Callable[[Any], Any] | None = None,
        name: str = "generic",
        dps: int = VERIFICATION_DPS,
    ) -> None:
        self._f = f
        self._fprime = fprime
        self.name = name
        self.dps = dps
        self.cost = CostCounter()

        # SolveResult records (e, M) for every solve; there is no eccentricity
        # here, so they are nan and every downstream consumer must ignore them.
        self.e = float("nan")
        self.M = float("nan")

    # -- evaluations ---------------------------------------------------
    def f(self, x):
        self.cost.note_point(x)
        self.cost.sin_only += 1
        return self._f(x)

    def fprime(self, x):
        self.cost.note_point(x)
        self.cost.cos_only += 1
        return self._derivative(x, 1)

    def f_fprime(self, x):
        self.cost.note_point(x)
        self.cost.sincos_pairs += 1
        return self._f(x), self._derivative(x, 1)

    def derivatives(self, x, order: int = 3):
        if order < 1 or order > 3:
            raise ValueError("order must be 1, 2 or 3")
        self.cost.note_point(x)
        self.cost.sincos_pairs += 1
        out = [self._f(x)]
        out.extend(self._derivative(x, k) for k in range(1, order + 1))
        return tuple(out)

    def _derivative(self, x, k: int):
        """f^(k)(x): analytic when supplied, otherwise from mpmath.

        ``mp.diff`` is numerical, but at the working precision used here its
        error is far below anything the order estimator can resolve.  Supply
        ``fprime`` explicitly when you have it.
        """
        if k == 1 and self._fprime is not None:
            return self._fprime(x)
        return mp.diff(self._f, x, k)

    # -- bookkeeping ---------------------------------------------------
    def reset_cost(self) -> None:
        self.cost.reset()

    def __repr__(self) -> str:
        return f"GenericProblem({self.name!r}, dps={self.dps})"


def refine_root(
    f: Callable[[Any], Any],
    approximate_root: float,
    dps: int = VERIFICATION_DPS,
) -> Any:
    """Sharpen a published root to ``dps`` digits.

    Uses mpmath's secant iteration from the paper's value.  Verifies the
    residual afterwards and raises rather than returning a root that only
    looks converged - a wrong reference root would silently corrupt every
    error, and therefore every order estimate, computed against it.
    """
    with mp.workdps(dps):
        root = mp.findroot(f, mp.mpf(approximate_root))
        residual = abs(f(root))
        # Generous relative to the precision (dps digits) but far tighter
        # than anything the order estimator can resolve.
        threshold = mp.mpf(10) ** (-(dps // 2))
        if residual > threshold:
            raise ValueError(
                f"refined root {mp.nstr(root, 12)} has residual "
                f"{mp.nstr(residual, 6)}, above {mp.nstr(threshold, 6)}; the "
                "published starting value or the transcribed function is wrong"
            )
        return +root


def verify_order_on_test_functions(
    solver_name: str,
    n_iterations: int = 6,
    dps: int = VERIFICATION_DPS,
    rel_tolerance: float = 0.05,
    functions: Sequence[tuple[str, Callable[[Any], Any], float, float]] | None = None,
) -> list[dict]:
    """Measure the empirical order of one solver on the paper's test set.

    Returns one record per test function with keys ``solver``, ``function``,
    ``measured_order``, ``claimed_order``, ``abs_diff``, ``passed``, plus the
    diagnostics needed to explain a failure.

    ``rel_tolerance`` is how far the measured order may sit from the claimed
    one and still pass.  5% is loose enough to absorb estimator noise and
    tight enough to catch a method that has silently degraded to its
    memoryless base (8 vs 8.8989 is an 11% gap; 8 vs 10.7446 is 26%).

    Runs with ``tol=0.0`` so the loop cannot exit early - the estimator needs
    the whole residual sequence, not just the part before convergence.
    """
    if functions is None:
        functions = PAPER_TEST_FUNCTIONS

    solver = get_solver(solver_name)
    claimed = solver.theoretical_order
    rows: list[dict] = []

    with mp.workdps(dps):
        for name, f, published_root, x0 in functions:
            root = refine_root(f, published_root, dps=dps)
            problem = GenericProblem(f, name=name, dps=dps)

            failure: str | None = None
            try:
                result = solver.solve(
                    problem,
                    E0=mp.mpf(x0),
                    tol=mp.mpf(0),
                    max_iter=n_iterations,
                    record_history=True,
                    E_reference=root,
                )
            except ArithmeticError as exc:
                # OverflowError and ZeroDivisionError both land here.  A
                # NotImplementedError from an unwritten solver deliberately
                # does NOT - it must propagate so the test suite skips.
                rows.append({
                    "solver": solver_name, "function": name,
                    "measured_order": float("nan"), "claimed_order": claimed,
                    "abs_diff": float("nan"), "passed": False,
                    "n_usable": 0, "iterations": 0,
                    "reason": f"solve raised {type(exc).__name__}: {exc}",
                })
                continue

            estimate = order_from_history_detail(result.history, use_reference=True)
            measured = estimate.value

            if claimed is None:
                passed, abs_diff, failure = True, float("nan"), "closed form: no asymptotic order to check"
            elif estimate.ok():
                abs_diff = abs(float(measured) - float(claimed))
                passed = abs_diff <= rel_tolerance * abs(float(claimed))
                failure = estimate.reason
            else:
                abs_diff, passed, failure = float("nan"), False, estimate.reason

            rows.append({
                "solver": solver_name,
                "function": name,
                "measured_order": float(measured) if estimate.ok() else float("nan"),
                "claimed_order": claimed,
                "abs_diff": abs_diff,
                "passed": passed,
                "n_usable": estimate.n_usable,
                "iterations": result.iterations,
                "reason": failure,
            })

    return rows


def _kepler_callables(e, M):
    """f(E) = E - e sin E - M and its derivative, in mpmath arithmetic."""
    def f(E):
        return E - e * mp.sin(E) - M

    def fprime(E):
        return 1 - e * mp.cos(E)

    return f, fprime


def measure_order_on_kepler(
    solver_name: str,
    points: Sequence[tuple[float, float]],
    n_iterations: int = 8,
    dps: int = VERIFICATION_DPS,
) -> list[dict]:
    """Measure each solver's order on KEPLER'S EQUATION, at extended precision.

    Distinct from :func:`verify_order_on_test_functions`, which uses the
    papers' generic test functions to check the transcription.  This asks the
    question the project actually cares about: does the claimed order survive
    contact with Kepler's own (e, M) operating range, including the corner
    where f'(E) = 1 - e cos E collapses towards zero?

    It must run in extended precision for the same reason the paper check
    does - an order-10 method exhausts double precision in two iterations, so
    a double-precision measurement here returns noise, not an order.  That is
    not a limitation worth working around; it is a finding worth reporting.

    The reference root is obtained by bracketing bisection in mpmath, never
    by one of the solvers under test.  |E - M| <= e < 1 for the elliptical
    case, so [M - 1.1, M + 1.1] always brackets it.
    """
    solver = get_solver(solver_name)
    claimed = solver.theoretical_order
    rows: list[dict] = []

    with mp.workdps(dps):
        for e_raw, M_raw in points:
            e, M = mp.mpf(e_raw), mp.mpf(M_raw)
            f, fprime = _kepler_callables(e, M)

            lo, hi = M - mp.mpf("1.1"), M + mp.mpf("1.1")
            if f(lo) * f(hi) > 0:
                raise ValueError(f"bad bracket at e={e_raw}, M={M_raw}")
            root = mp.findroot(f, (lo, hi), solver="anderson")

            problem = GenericProblem(f, fprime, name=f"kepler_e{e_raw}", dps=dps)
            try:
                result = solver.solve(
                    problem, E0=M, tol=mp.mpf(0), max_iter=n_iterations,
                    record_history=True, E_reference=root,
                )
            except ArithmeticError as exc:
                rows.append({"solver": solver_name, "e": float(e_raw),
                             "M": float(M_raw), "claimed_order": claimed,
                             "measured_order": float("nan"), "n_usable": 0,
                             "reason": f"{type(exc).__name__}: {exc}"})
                continue

            estimate = order_from_history_detail(result.history, use_reference=True)
            rows.append({
                "solver": solver_name,
                "e": float(e_raw),
                "M": float(M_raw),
                "claimed_order": claimed,
                "measured_order": float(estimate.value) if estimate.ok() else float("nan"),
                "n_usable": estimate.n_usable,
                "reason": estimate.reason,
            })

    return rows


#: A converged iterate further than this from the reference root is on a
#: DIFFERENT root, not a slightly inaccurate one.  Kepler's branches are
#: separated by order 2*pi, and a correct solve lands within ~1e-13, so
#: anything in between is unambiguous.  Deliberately far above numerical
#: noise and far below the branch spacing.
BRANCH_TOL = 1e-6


def kepler_check_points(n_points: int = 5000, seed: int = 0
                        ) -> list[tuple[float, float]]:
    """(e, M) sample covering the full range, including the hard corner.

    Reuses the grid module rather than rolling a fresh sampler, so the
    correctness check visits the same kind of territory the benchmark does.
    Roughly half ordinary operating range, a quarter pathological corner, a
    quarter RadVel-realistic - the corner is over-weighted relative to its
    area on purpose, because that is where a solver jumps branches.

    ``n_points`` is a target; the blocks are sized to land near it.
    """
    quarter = max(1, n_points // 4)
    side = max(2, int(round(math.sqrt(2 * quarter))))
    corner = max(2, int(round(math.sqrt(quarter))))
    return (uniform_grid(n_e=side, n_M=side, e_max=0.99)
            + pathological_grid(n_e=corner, n_M=corner)
            + radvel_operating_grid(n_samples=quarter, seed=seed))


def verify_kepler_correctness(solver_name: str, n_points: int = 5000,
                              tol: float = 1e-13, max_iter: int = 100,
                              guess_name: str = "simple",
                              seed: int = 0) -> dict:
    """Check the solver returns the RIGHT root on Kepler, not just a root.

    Two failure modes, kept apart because they mean opposite things:

    * **Non-convergence** is honest.  A solver that reports failure in the
      pathological corner has told the truth about itself, and the
      robustness metric is where that belongs - not here.  It does not fail
      this check.
    * **A wrong root** is not honest.  The iteration jumped to another
      branch, the residual |f(E)| is tiny because that branch really is a
      root, ``converged`` is True, and every downstream number computed from
      E is silently wrong.  This is the failure this function exists to
      catch, and the only one that fails it.

    Compared against :func:`reference.reference_root`, which is an
    independent bracketing solve at 50 digits - never one of the methods
    under test.

    Returns a dict with the counts, the worst offender, and ``passed``.
    """
    solver = get_solver(solver_name)
    guess = get_guess(guess_name)
    points = kepler_check_points(n_points, seed=seed)

    n_converged = n_failed = n_wrong = 0
    max_error = 0.0
    worst: tuple[float, float] | None = None
    worst_wrong: tuple[float, float] | None = None

    for e, M in points:
        reference = reference_root(e, M)
        result = solve_one(solver, guess, e=e, M=M, tol=tol,
                           max_iter=max_iter, E_reference=reference)

        # ``converged`` is the only honest gate. Falling back to
        # "no exception was raised" would count a run that simply exhausted
        # max_iter as an answer: Newton at e=0.9999, M=0.1 walks off to
        # E = 8.8e12 with a residual of the same size, and would then be
        # reported as a WRONG ROOT rather than as the non-convergence it is.
        # Markley needs no special case here - it is closed-form but still
        # reports converged=True with iterations=0.
        if not result.converged:
            n_failed += 1
            continue

        n_converged += 1
        error = abs(result.E - reference)
        if error > BRANCH_TOL:
            n_wrong += 1
            if worst_wrong is None:
                worst_wrong = (e, M)
        elif error > max_error:
            max_error, worst = error, (e, M)

    total = len(points)
    return {
        "solver": solver_name,
        "check": "kepler_correctness",
        "n_points": total,
        "n_converged": n_converged,
        "n_failed": total - n_converged,
        "n_wrong_root": n_wrong,
        "fraction_wrong_root": n_wrong / total if total else float("nan"),
        "fraction_converged": n_converged / total if total else float("nan"),
        # Max error over the points that found the RIGHT root; mixing the
        # wrong-root points in here would report ~2*pi and hide the real
        # accuracy.
        "max_abs_error": max_error,
        "worst_point": worst,
        "first_wrong_point": worst_wrong,
        "passed": n_wrong == 0,
    }


EXPERIMENT = "verification"

#: Column order for the two tables. Stated explicitly so that a run which
#: skips every solver still writes a file with a header rather than a
#: zero-byte one that pandas refuses to read back - the same guarantee
#: io.results_io makes for the raw tables.
ORDER_COLUMNS = ("solver", "function", "measured_order", "claimed_order",
                 "abs_diff", "passed", "n_usable", "iterations", "reason")
SUMMARY_COLUMNS = (
    "solver", "claimed_order", "measured_order_mean", "measured_order_min",
    "measured_order_max", "n_functions", "n_functions_passed",
    "order_check_passed", "kepler_n_points", "kepler_fraction_converged",
    "kepler_n_wrong_root", "kepler_max_abs_error", "kepler_check_passed",
    "passed",
)


def _table(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    """Rows to a DataFrame with a stable column order, header even if empty."""
    if not rows:
        return pd.DataFrame(columns=list(columns))
    frame = pd.DataFrame(rows)
    present = [c for c in columns if c in frame.columns]
    extra = [c for c in frame.columns if c not in present]
    return frame[present + extra]


def run(config_path: str) -> None:
    """Entry point used by scripts/run_verification.py.

    Runs both checks for every solver in the config and writes

      results/verification/order.csv     one row per (solver, test function)
      results/verification/summary.csv   one row per solver, both checks

    then prints a PASS/FAIL line each.  Every number in those files is
    measured here; nothing is transcribed by hand.

    A solver that is still a skeleton is warned about and skipped rather
    than aborting the run, so this stays usable while the team works in
    parallel.
    """
    cfg = load_config(config_path)
    dps = int(cfg.extra.get("working_precision_dps", VERIFICATION_DPS))
    n_points = int(cfg.extra.get("kepler_check_points", 5000))
    # cfg.tol is 0.0 for this experiment - that is what forces the order run
    # to take exactly max_iter iterations. It is NOT a usable stopping
    # tolerance for the correctness sweep, which needs a real one.
    kepler_tol = float(cfg.extra.get("kepler_check_tol", 1e-13))
    guess_name = cfg.guesses[0] if cfg.guesses else "simple"

    print(f"verification: {len(cfg.solvers)} solvers, {dps} dps, "
          f"{len(PAPER_TEST_FUNCTIONS)} test functions, "
          f"~{n_points} Kepler points")

    order_rows: list[dict] = []
    summary_rows: list[dict] = []

    for solver_name in cfg.solvers:
        try:
            rows = verify_order_on_test_functions(
                solver_name, n_iterations=cfg.max_iter, dps=dps)
            correctness = verify_kepler_correctness(
                solver_name, n_points=n_points, tol=kepler_tol,
                guess_name=guess_name, seed=cfg.seed)
        except NotImplementedError as exc:
            warnings.warn(f"skipping {solver_name}: {exc}", stacklevel=2)
            continue

        order_rows.extend(rows)
        measured = [r["measured_order"] for r in rows
                    if r["measured_order"] == r["measured_order"]]
        order_passed = all(r["passed"] for r in rows)

        summary_rows.append({
            "solver": solver_name,
            "claimed_order": rows[0]["claimed_order"] if rows else None,
            "measured_order_mean": sum(measured) / len(measured) if measured else float("nan"),
            "measured_order_min": min(measured) if measured else float("nan"),
            "measured_order_max": max(measured) if measured else float("nan"),
            "n_functions": len(rows),
            "n_functions_passed": sum(1 for r in rows if r["passed"]),
            "order_check_passed": order_passed,
            "kepler_n_points": correctness["n_points"],
            "kepler_fraction_converged": correctness["fraction_converged"],
            "kepler_n_wrong_root": correctness["n_wrong_root"],
            "kepler_max_abs_error": correctness["max_abs_error"],
            "kepler_check_passed": correctness["passed"],
            "passed": order_passed and correctness["passed"],
        })

    order_path = results_path(EXPERIMENT, "order.csv")
    _table(order_rows, ORDER_COLUMNS).to_csv(order_path, index=False)
    summary_path = results_path(EXPERIMENT, "summary.csv")
    _table(summary_rows, SUMMARY_COLUMNS).to_csv(summary_path, index=False)

    print()
    for row in summary_rows:
        verdict = "PASS" if row["passed"] else "FAIL"
        claimed = row["claimed_order"]
        claimed_text = "closed form" if claimed is None else f"{claimed:.4f}"
        print(
            f"  {verdict}  {row['solver']:<8s} "
            f"order {row['measured_order_mean']:.4f} vs {claimed_text} "
            f"({row['n_functions_passed']}/{row['n_functions']} functions)   "
            f"kepler {row['kepler_n_wrong_root']} wrong root(s), "
            f"max err {row['kepler_max_abs_error']:.2e}, "
            f"{row['kepler_fraction_converged']:.1%} converged"
        )

    n_failed = sum(1 for row in summary_rows if not row["passed"])
    print()
    print(f"wrote {order_path}")
    print(f"wrote {summary_path}")
    if n_failed:
        # Loud, because the project's rule is that a solver does not enter
        # the grid benchmark until this passes.
        print(f"\n{n_failed} solver(s) FAILED verification - do not run the "
              "grid benchmark until this is resolved.")
