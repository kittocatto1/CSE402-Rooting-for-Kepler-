"""Work Plan step 2 - verify each implementation before trusting it.

The rule for this project: a solver does not enter the grid benchmark until
its MEASURED convergence order matches what its source paper claims, on the
paper's OWN test functions.  This is what stops a transcription bug from
being reported as "the with-memory method underperforms".

Owner: Suchi.
"""

from __future__ import annotations

from typing import Callable

#: Test functions from the two with-memory papers. Each entry is
#: (name, f, known_root, x0). Fill these in from the papers' numerical
#: examples sections - use the SAME functions and the SAME starting points,
#: otherwise the comparison against their reported order is meaningless.
PAPER_TEST_FUNCTIONS: list[tuple[str, Callable[[float], float], float, float]] = [
    # TODO(Suchi): e.g. ("f1", lambda x: x**3 + 4*x**2 - 10, 1.365230013..., 1.5)
]


def verify_order_on_test_functions(solver_name: str, n_iterations: int = 6) -> dict:
    """Measure the empirical order of one solver on the paper's test set.

    TODO(Suchi):
      1. The solvers in this package are written against KeplerProblem, not
         against a generic f. Add a thin adapter so a solver can be driven by
         an arbitrary f for verification only - either a GenericProblem class
         with the same evaluation methods, or a KeplerProblem subclass that
         overrides f/f_fprime/derivatives. Put it in this module, NOT in
         core/, so the benchmark path stays Kepler-specific.
      2. Run with tol=0.0 and max_iter=n_iterations so the full residual
         sequence is produced (no early exit).
      3. Feed the history to evaluation.convergence_order.coc / acoc.
      4. Return {"solver", "function", "measured_order", "claimed_order",
                 "abs_diff", "passed"} per test function.
      5. Use high precision if needed - at order 10 you hit double-precision
         floor after ~3 iterations, so the order estimate has very few usable
         points. Consider running this step with mpmath at 100 digits. This
         is the single most common reason an order check "fails" when the
         code is actually right.
    """
    raise NotImplementedError("verify_order_on_test_functions: see TODO above")


def verify_kepler_correctness(solver_name: str, n_points: int = 5000,
                              tol: float = 1e-13) -> dict:
    """Check the solver returns the RIGHT root on Kepler, not just a root.

    TODO(Suchi):
      1. Sample (e, M) over the full range.
      2. Solve, and compare against reference.reference_root.
      3. Report max |E - E_ref| and the fraction of points where the solver
         converged to something that is not the reference root - the second
         failure mode is easy to miss because the residual looks fine when
         the iteration has jumped to a different branch.
    """
    raise NotImplementedError("verify_kepler_correctness: see TODO above")


def run(config_path: str) -> None:
    """Entry point used by scripts/run_verification.py.

    TODO(Suchi): load the config, run both checks for every solver listed,
    write the table to results/verification/summary.csv, and print a clear
    PASS/FAIL line per solver. Do not write any numbers by hand.
    """
    raise NotImplementedError("verification.run: see TODO above")
