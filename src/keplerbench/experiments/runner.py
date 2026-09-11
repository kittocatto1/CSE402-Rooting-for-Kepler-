"""The common pipeline: one function that runs any solver with any guess.

This is the fairness guarantee of the whole project.  Every solver goes
through ``solve_one``, so every solver gets the same stopping rule, the same
cost instrumentation and the same reference root.  No experiment may call a
solver directly.

Owner: Mahdi.
"""

from __future__ import annotations

import time
import warnings
from statistics import median
from typing import Iterable, Sequence

from keplerbench.core.base import InitialGuess, KeplerSolver
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.core.types import KeplerProblem, SolveResult

#: Solvers that build their answer from a closed form and ignore E0.
#: Run once under the pseudo-guess label "n/a" - see run_sweep.
GUESS_INDEPENDENT = {"markley"}


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
    progress_every: int = 500,
) -> list[SolveResult]:
    """Cross every solver with every guess over every (e, M) point.

    A solver whose ``step`` is still a skeleton raises NotImplementedError;
    that combination is warned about once and skipped, so the sweep keeps
    working while the team is still writing solvers in parallel.

    Closed-form solvers (see ``GUESS_INDEPENDENT``) ignore the starting
    guess, so they are run ONCE under the pseudo-guess label "n/a" instead of
    once per guess - otherwise the tables carry four identical rows.
    """
    solvers = {name: get_solver(name) for name in solver_names}
    guesses = {name: get_guess(name) for name in guess_names}
    if not guesses:
        raise ValueError("run_sweep needs at least one guess")
    # Closed-form solvers still need *some* E0 to satisfy the interface.
    fallback_guess = next(iter(guesses.values()))

    results: list[SolveResult] = []
    broken: set[tuple[str, str]] = set()

    for i, (e, M) in enumerate(points, start=1):
        E_reference = None if reference_roots is None else reference_roots.get((e, M))
        for solver_name, solver in solvers.items():
            if solver_name in GUESS_INDEPENDENT:
                pairs = [("n/a", fallback_guess)]
            else:
                pairs = list(guesses.items())
            for guess_label, guess in pairs:
                key = (solver_name, guess_label)
                if key in broken:
                    continue
                try:
                    r = solve_one(
                        solver, guess, e, M,
                        tol=tol,
                        max_iter=max_iter,
                        record_history=record_history,
                        E_reference=E_reference,
                    )
                except NotImplementedError as exc:
                    broken.add(key)
                    warnings.warn(
                        f"skipping {solver_name}+{guess_label}: not implemented yet ({exc})",
                        stacklevel=2,
                    )
                    continue
                r.guess = guess_label
                results.append(r)
        if progress_every and i % progress_every == 0:
            print(f"  run_sweep: {i} points, {len(results)} rows", flush=True)

    return results


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

    Wall-clock is MACHINE-DEPENDENT and must be reported as such; the cost
    counters of Section 4.1 are the machine-independent measure and stay the
    headline number.  History recording is forced off - it allocates per
    iteration and would dominate the timing.
    """
    warmup = max(1, repeats // 100)
    for _ in range(warmup):
        solve_one(solver, guess, e, M, tol=tol, max_iter=max_iter)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        solve_one(solver, guess, e, M, tol=tol, max_iter=max_iter)
        samples.append(time.perf_counter_ns() - t0)
    # Median, not mean: one OS hiccup must not decide which solver wins.
    return median(samples) * 1e-9
