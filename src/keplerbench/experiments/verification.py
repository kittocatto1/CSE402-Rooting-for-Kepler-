"""Work Plan step 2 - check every solver before trusting it.

The rule: a solver does not go into the grid benchmark until its measured
convergence order matches what its own paper claims, on that paper's own
test functions. This is what stops a typing mistake in a formula from being
written up as "the new method is slow".

Owner: Suchi.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
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
# _write_meta is private; see the note in _save_table for why it is used here
# and why it should be promoted to the public API.
from keplerbench.io.results_io import _write_meta, results_path
from keplerbench.reference.mpmath_reference import reference_root

__all__ = [
    "BRANCH_TOL",
    "kepler_check_points",
    "measure_order_on_kepler",
    "GenericProblem",
    "PAPER_TEST_FUNCTIONS",
    "NWM9_TEST_FUNCTIONS",
    "SOLVER_TEST_FUNCTIONS",
    "VERIFICATION_DPS",
    "refine_root",
    "verify_order_on_test_functions",
    "verify_kepler_correctness",
    "run",
]

#: How many decimal digits to work in. Matches
#: extra.working_precision_dps in configs/verification.yaml.
#:
#: This looks absurdly high, and the reason is worth knowing. An order-p
#: method multiplies its correct digits by p each step: starting at 1e-1, an
#: order-9 method passes 1e-9, then 1e-81, then 1e-729. The estimator needs
#: three usable errors in a row to produce ONE number, so the precision has
#: to outrun p**3 digits or the sequence runs out before that happens.
#:
#: Calibrated on NWM9 (claimed 8.8989) over the NWM11 paper's eight test
#: functions (PAPER_TEST_FUNCTIONS), before NWM9 had its own set:
#:
#:     dps=100    measured 8.87-10.19, only 3 usable terms  -> FAILS
#:     dps=1000   measured 8.87-9.04                        -> 0.55 s
#:     dps=2000   measured 8.87-8.98                        -> 2.14 s
#:     dps=3000   identical to dps=2000                     -> 4.70 s
#:
#: 2000 is where the estimate stops improving. On NWM9's own eight functions
#: (NWM9_TEST_FUNCTIONS) the committed run gives 8.885-9.000
#: (results/verification/order.csv).  NWM11 (order 10.7446) burns
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
#: whatever precision the active context is using.  NWM9 has its own set,
#: below, because it comes from a different paper.
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

#: NWM9's own test set: the numerical examples of Mittal, Panday & Jantschi,
#: Mathematics 12(22), 3490, 2024 (doi:10.3390/math12223490), Section 3.
#:
#: Kept separate from PAPER_TEST_FUNCTIONS above rather than merged, because
#: "validated against its source paper's reported order on the paper's OWN
#: test set" is the project's rule, and the two papers do not share a set.
#: Measuring NWM9 on NWM11's functions is still evidence - order is a
#: property of the method, not of the function - but it does not reproduce
#: the table the paper actually published.
#:
#: SIX of these eight needed their superscripts reconstructed: the PDF prints
#: them on the line ABOVE the formula, so zeta3 extracts as x*e^x when the
#: paper means x*e^(x^2), and zeta1's e^(cos(x/2)) extracts as e^(cos x)/2.
#: Every reading below was checked by confirming the paper's published root
#: actually zeroes it - the same discipline used for the set above, and the
#: only reason the errors were caught.
#:
#: zeta8 is the civil-engineering beam model of the paper's Example 8. Its
#: quartic has a DOUBLE root at x = 2, but the paper's starting point
#: x0 = -0.55 converges to the simple root -4 + 2*sqrt(3) = -0.53589..., so
#: the order theory still applies. Starting nearer 2 would measure a
#: different (lower) order, which is expected behaviour at a multiple root
#: and not a defect.
NWM9_TEST_FUNCTIONS: list[tuple[str, Callable[[Any], Any], float, float]] = [
    ("zeta1", lambda x: 1 + x**2 * mp.e ** mp.cos(x / 2)
                        - (x + 1) * mp.e ** mp.sin(x / 2), 0.8475, 0.9),
    ("zeta2", lambda x: mp.e ** (x**3 + mp.cos(x) + 1) - x**2 + x + 1,
     -1.0787, -0.8),
    ("zeta3", lambda x: x * mp.e ** (x**2) - mp.sin(x) ** 2
                        + 3 * mp.cos(x) + 5, -1.2076, -1.2),
    ("zeta4", lambda x: mp.e ** (-(x**2)) * (1 + x**3 + x**6) * (x - 2),
     2.0000, 1.95),
    ("zeta5", lambda x: x**7 - 4 * x**4 + x - 1, 1.5749, 1.58),
    ("zeta6", lambda x: mp.e ** (x**2 - 4) + mp.sin(x - 2) - x**4 + 15,
     2.0000, 2.1),
    ("zeta7", lambda x: mp.e ** (-(x**2) + x + 2) - 1, 2.0000, 2.01),
    ("zeta8", lambda x: x**4 + 4 * x**3 - 24 * x**2 + 16 * x + 16,
     -0.5358983848622454, -0.55),
]

#: Which set a solver is verified against. Anything absent falls back to
#: PAPER_TEST_FUNCTIONS, which is right for NWM10/NWM11 and harmless for the
#: classical methods - Newton and Danby have no "own paper test set" in this
#: project's sense, and their orders are textbook.
SOLVER_TEST_FUNCTIONS: dict[str, list] = {
    "nwm9": NWM9_TEST_FUNCTIONS,
}


# ----------------------------------------------------------------------
# Driving a Kepler solver with an arbitrary f
# ----------------------------------------------------------------------
class GenericProblem:
    """Lets a solver run on any function, for verification only.

    Our solvers are written to work on a KeplerProblem, but the papers
    report their orders on ordinary test functions. This offers the same
    set of methods, so the normal solve loop can drive a solver unchanged.

    It lives here rather than in ``core/`` on purpose: the benchmark itself
    should stay Kepler-only, and nothing on that path should be able to
    import a generic problem by accident.

    A note on the cost counters
    ---------------------------
    They are still incremented, so solvers that report interpolated
    derivatives keep working. But on a plain function ``sincos_pairs`` just
    means "asked for value and slope together" and ``sin_only`` means
    "asked for the value". There is no sin or cos here, so these numbers
    must never end up in the Section 4.1 cost tables.
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

        # Every SolveResult carries (e, M). There is no orbit here, so these
        # are nan and nothing downstream should read them.
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
        """The k-th derivative of f at x.

        Uses the exact derivative when one was supplied, otherwise asks
        mpmath to work it out numerically. mpmath's version is approximate,
        but at the precision used here its error is far below anything the
        order estimator can see. Pass ``fprime`` if you have it.
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
    """Sharpen a root printed in a paper to ``dps`` digits.

    Starts from the paper's value and refines it with mpmath. Afterwards it
    checks the residual and raises if it is too large, rather than handing
    back something that only looks converged. A wrong reference root would
    quietly poison every error, and so every order estimate, measured
    against it.
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
    """Measure one solver's order on its paper's own test functions.

    Returns one row per test function: the solver, the function, the order
    measured, the order claimed, the gap between them, whether it passed,
    and enough detail to explain a failure.

    ``rel_tolerance`` is how far the measured order may be from the claimed
    one and still count as a pass. 5% is loose enough to absorb noise in the
    estimate, and tight enough to catch a method that has quietly dropped
    back to its base version: 8 against 8.8989 is an 11% gap, 8 against
    10.7446 is 26%.

    Runs with ``tol=0.0`` so the loop cannot stop early. The estimator needs
    the whole run of residuals, not just the part before it converged.
    """
    if functions is None:
        # Each solver against its OWN paper's examples where they differ.
        functions = SOLVER_TEST_FUNCTIONS.get(solver_name, PAPER_TEST_FUNCTIONS)

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
    """Measure order on Kepler's equation itself, at high precision.

    Different question from :func:`verify_order_on_test_functions`. That one
    uses the papers' own functions to check we transcribed the formulas
    correctly. This one asks what the project actually cares about: does the
    claimed order hold up on Kepler, across the whole range of (e, M),
    including the corner where f'(E) = 1 - e cos E drops towards zero?

    Needs high precision for the same reason the other check does. An
    order-10 method uses up a double in two steps, so measuring here in
    ordinary precision returns noise rather than an order. That is not a
    problem to work around - it is a result worth reporting.

    The reference root comes from bracketing in mpmath, never from one of
    the solvers being tested. For an elliptical orbit |E - M| <= e < 1, so
    [M - 1.1, M + 1.1] always contains the root.
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
    """A spread of (e, M) points covering the full range and the hard corner.

    Reuses the grid module instead of writing a new sampler, so this check
    visits the same kind of ground the benchmark does. Roughly half ordinary
    orbits, a quarter in the hard corner, a quarter realistic RadVel values.
    The corner gets more points than its area deserves, on purpose: that is
    where a solver is most likely to land on the wrong root.

    ``n_points`` is a target, not exact - the blocks are sized to land near
    it.
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
    """Check the solver finds the RIGHT root on Kepler, not just any root.

    Two kinds of failure, kept apart because they mean opposite things:

    * **Not converging is honest.** A solver that gives up in the hard
      corner has told the truth about itself. That belongs in the
      robustness numbers, not here, so it does not fail this check.
    * **A wrong root is not honest.** The iteration has jumped to a
      different branch. |f(E)| is tiny, because that branch really is a
      root, and the converged flag is True - but every number computed from
      E afterwards is wrong. That is what this function exists to catch,
      and the only thing that fails it.

    Compared against :func:`reference.reference_root`, a separate 50-digit
    bracketing solve - never one of the methods being tested.

    Returns the counts, the worst offender, and whether it passed.
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
#: Eccentricities at which the order is measured on Kepler's equation
#: itself, for the order-versus-eccentricity figure. Weighted towards
#: e -> 1: the whole question is whether a claimed order survives the
#: pathological corner, and a linear sweep spends almost all its points
#: where nothing interesting happens. Overridable via
#: ``extra.eccentricity_scan``.
ECCENTRICITY_SCAN = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999)

#: Mean anomaly for that scan. Away from 0 so the scan isolates the effect
#: of eccentricity rather than mixing in the M -> 0 stiffness.
ECCENTRICITY_SCAN_M = 0.3

KEPLER_ORDER_COLUMNS = ("solver", "e", "M", "measured_order", "claimed_order",
                        "n_usable", "reason")

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


def _save_table(rows: list[dict], columns: tuple[str, ...], filename: str,
                config) -> Path:
    """Write one table plus the provenance sidecar that belongs to it.

    The house rule in ``io/results_io.py`` is that every table in the report
    can be traced back to the settings and the commit that produced it.
    ``save_results`` does that automatically, but only for tables made of
    :class:`SolveResult` objects - these two are per-solver summaries, so the
    sidecar has to be written explicitly.

    NOTE(Suchi -> Anisa): this is the only place in the project that uses
    ``results_io._write_meta`` despite its leading underscore. Copying the
    format here instead would be worse, since we would end up with two
    sidecar layouts in one results/ folder. Better still would be to make
    the helper public: error_propagation.py writes summary tables the same
    way and has the same gap.
    """
    table = _table(rows, columns)
    path = results_path(EXPERIMENT, filename)
    table.to_csv(path, index=False)
    _write_meta(path, EXPERIMENT, len(table), config)
    return path


def run(config_path: str) -> None:
    """What scripts/run_verification.py calls.

    Runs both checks on every solver listed in the config, then writes

      results/verification/order.csv     one row per solver per function
      results/verification/summary.csv   one row per solver, both checks

    and prints a PASS or FAIL line for each. Every number in those files is
    measured here - none of it is typed in by hand.

    A solver that is not written yet gets a warning and is skipped, rather
    than stopping the whole run. That keeps this usable while the team is
    still working in parallel.
    """
    cfg = load_config(config_path)
    dps = int(cfg.extra.get("working_precision_dps", VERIFICATION_DPS))
    n_points = int(cfg.extra.get("kepler_check_points", 5000))
    # cfg.tol is 0.0 here on purpose: that is what makes the order run take
    # exactly max_iter steps. It is NOT a usable stopping tolerance for the
    # correctness sweep below, which needs a real one.
    kepler_tol = float(cfg.extra.get("kepler_check_tol", 1e-13))
    guess_name = cfg.guesses[0] if cfg.guesses else "simple"

    print(f"verification: {len(cfg.solvers)} solvers, {dps} dps, "
          f"{len(PAPER_TEST_FUNCTIONS)} test functions "
          f"({len(NWM9_TEST_FUNCTIONS)} for nwm9, from its own paper), "
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

    # Order measured on KEPLER itself across the eccentricity range. This is
    # a different question from the paper test functions above - those check
    # the transcription, this checks whether the claimed order survives the
    # equation we actually care about - and it is what feeds the
    # order-versus-eccentricity figure. Nothing else persists it.
    scan = [float(e) for e in cfg.extra.get("eccentricity_scan",
                                            ECCENTRICITY_SCAN)]
    scan_M = float(cfg.extra.get("eccentricity_scan_M", ECCENTRICITY_SCAN_M))
    kepler_rows: list[dict] = []
    for solver_name in [row["solver"] for row in summary_rows]:
        try:
            kepler_rows.extend(measure_order_on_kepler(
                solver_name, [(e, scan_M) for e in scan],
                n_iterations=max(cfg.max_iter, 10), dps=dps))
        except (NotImplementedError, ArithmeticError) as exc:
            warnings.warn(
                f"eccentricity scan: skipping {solver_name}: {exc}",
                stacklevel=2,
            )

    order_path = _save_table(order_rows, ORDER_COLUMNS, "order.csv", cfg)
    summary_path = _save_table(summary_rows, SUMMARY_COLUMNS, "summary.csv", cfg)
    kepler_path = _save_table(kepler_rows, KEPLER_ORDER_COLUMNS,
                              "kepler_order.csv", cfg)

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
    print(f"wrote {kepler_path}")
    if n_failed:
        # Loud, because the project's rule is that a solver does not enter
        # the grid benchmark until this passes.
        print(f"\n{n_failed} solver(s) FAILED verification - do not run the "
              "grid benchmark until this is resolved.")


def reached_root_within(solver_name: str, e: float, M: float, n_iterations: int = 8) -> bool:
    """Does the solver reach the root from E0 = M within ``n_iterations``?

    The order scan on Kepler's equation runs ten iterations from E0 = M. If
    the solver has not reached the root by the eighth, the last few errors
    belong to a wandering iterate, and the ratio the order estimator forms
    from them is not a convergence order. Figure and table use this test to
    show such entries as "n.c." instead of a number.
    """
    import keplerbench.guesses  # noqa: F401
    import keplerbench.solvers  # noqa: F401
    from keplerbench.core.registry import get_guess, get_solver
    from keplerbench.experiments.runner import solve_one

    result = solve_one(get_solver(solver_name), get_guess("simple"), e, M,
                       max_iter=n_iterations)
    return bool(result.converged)
