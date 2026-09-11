"""The common pipeline: one function that runs any solver with any guess.

This is the fairness guarantee of the whole project.  Every solver goes
through ``solve_one``, so every solver gets the same stopping rule, the same
cost instrumentation and the same reference root.  No experiment may call a
solver directly.

Owner: Mahdi.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from keplerbench.core.base import InitialGuess, KeplerSolver
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.core.types import KeplerProblem, SolveResult


def solve_one(
    solver: KeplerSolver,
    guess: InitialGuess,
    e: float,
    M: float,
    tol: float = 1e-14,
    max_iter: int = 50,
    record_history: bool = False,
    E_reference: float | None = None,
) -> SolveResult:
    """Run one (solver, guess) pair on one (e, M).

    Fully implemented - this is the contract, do not fork it.
    """
    problem = KeplerProblem(e=e, M=M)
    E0 = guess(problem)
    problem.reset_cost()  # the guess must not be charged to the iteration cost

    result = solver.solve(
        problem,
        E0=E0,
        tol=tol,
        max_iter=max_iter,
        record_history=record_history,
        E_reference=E_reference,
    )
    result.guess = guess.name
    return result


def run_sweep(
    solver_names: Sequence[str],
    guess_names: Sequence[str],
    points: Iterable[tuple[float, float]],
    tol: float = 1e-14,
    max_iter: int = 50,
    record_history: bool = False,
    reference_roots: dict[tuple[float, float], float] | None = None,
) -> list[SolveResult]:
    """Cross every solver with every guess over every (e, M) point.

    TODO(Mahdi):
      1. Instantiate each solver and guess ONCE via get_solver / get_guess
         (not inside the loop - object creation would pollute the timing).
      2. Triple loop: for each point, for each guess, for each solver, call
         solve_one and collect the result.
      3. Look up the reference root from ``reference_roots`` when given.
      4. Catch NotImplementedError per (solver, guess) and skip that
         combination with a warning, so a half-finished solver does not stop
         the whole sweep while the team is still working in parallel.
      5. Show progress - the full grid is large; print every N points.
      6. Special case: Markley ignores the guess. Run it under a single
         pseudo-guess label "n/a" instead of once per guess, so the tables do
         not show five identical Markley rows. See README.
    """
    raise NotImplementedError("run_sweep: see TODO above")


def time_solve(
    solver: KeplerSolver,
    guess: InitialGuess,
    e: float,
    M: float,
    repeats: int = 1000,
    tol: float = 1e-14,
    max_iter: int = 50,
) -> float:
    """Median seconds per solve, measured properly.

    ``SolveResult.wall_time`` from a single solve is far too noisy to report.
    Use this for the wall-clock metric in Section 4.3.

    TODO(Mahdi):
      1. Warm up (a few untimed solves) so the first-call overhead and any
         caching are out of the way.
      2. Time ``repeats`` solves with time.perf_counter_ns, with
         record_history=False (recording allocates and would dominate).
      3. Return the MEDIAN per-solve time, not the mean - one OS hiccup
         should not decide which solver "wins".
      4. Note in the results that timing is machine-dependent; the cost
         counters from Section 4.1 are the machine-independent measure and
         should be the headline number in the report.
    """
    raise NotImplementedError("time_solve: see TODO above")
