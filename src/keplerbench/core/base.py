"""Abstract interfaces: one shape for every guess and every solver.

This is the "common interface" the proposal asks for.  Because all five
solvers and all four guesses expose exactly these methods, the experiment
runner can loop over them blindly and the comparison stays fair.

Owner: Anisa.  Everyone else *implements* against this file.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Any

from keplerbench.core.types import IterationRecord, KeplerProblem, SolveResult


# ----------------------------------------------------------------------
# Starting guesses  (Proposal Section 3.1, "initial-guess layer")
# ----------------------------------------------------------------------
class InitialGuess(ABC):
    """A rule that turns (e, M) into a first estimate E0.

    Kept separate from the solvers on purpose: the proposal wants "better
    start" and "better iteration" measured as two independent factors.
    """

    #: Short id used in configs, result files and plot legends.
    name: str = "unnamed-guess"
    #: One line for tables in the report.
    description: str = ""

    @abstractmethod
    def __call__(self, problem: KeplerProblem) -> float:
        """Return E0 for this (e, M). May use ``problem`` evaluations."""
        raise NotImplementedError


# ----------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------
class KeplerSolver(ABC):
    """Base class for anything that can return a root of Kepler's equation.

    Two kinds of solver subclass this:
      * iterative ones (Newton, Danby, NWM9, NWM11) -> use IterativeSolver
      * near-closed-form ones (Markley)             -> subclass this directly
    """

    name: str = "unnamed-solver"
    #: Theoretical convergence order claimed by the source paper.
    #: None for closed-form methods. We MEASURE this separately - see
    #: evaluation/convergence_order.py - and compare against this value.
    theoretical_order: float | None = None
    #: "baseline" | "established" | "with-memory" | "closed-form"
    category: str = "uncategorised"
    #: Literature reference, printed in the methods table of the report.
    reference: str = ""

    @abstractmethod
    def solve(
        self,
        problem: KeplerProblem,
        E0: float,
        tol: float = 1e-14,
        max_iter: int = 50,
        record_history: bool = False,
        E_reference: float | None = None,
    ) -> SolveResult:
        """Solve and return a fully-populated :class:`SolveResult`."""
        raise NotImplementedError


class IterativeSolver(KeplerSolver):
    """Standard iteration loop shared by every iterative method.

    Subclasses implement only the *update rule*.  The loop below handles
    history recording, the stopping test, cost bookkeeping and failure
    capture, so that no solver can accidentally get an unfair advantage by
    stopping on a different criterion.

    To write a new solver you override at most two methods::

        class MySolver(IterativeSolver):
            name = "mine"
            def init_state(self, problem, E0):
                return {}                       # only if you need memory
            def step(self, problem, E, state):
                f, fp = problem.f_fprime(E)
                return E - f / fp
    """

    #: Stopping rule: "residual" (|f| <= tol), "step" (|dE| <= tol), or "both".
    stopping_rule: str = "residual"

    def init_state(self, problem: KeplerProblem, E0: float) -> dict[str, Any]:
        """Per-solve scratch space.

        With-memory methods (NWM9 / NWM11) keep their recycled values and
        self-accelerating parameters in here.  Memoryless methods ignore it.
        """
        return {}

    @abstractmethod
    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        """One full iteration: given E_n, return E_{n+1}.

        Mutate ``state`` in place if the method carries memory.
        Evaluate the equation only through ``problem`` so cost is counted.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _converged(self, residual: float, step: float | None, tol: float) -> bool:
        """Apply the shared stopping rule."""
        if self.stopping_rule == "residual":
            return residual <= tol
        if self.stopping_rule == "step":
            return step is not None and step <= tol
        if self.stopping_rule == "both":
            return residual <= tol and step is not None and step <= tol
        raise ValueError(f"unknown stopping_rule {self.stopping_rule!r}")

    def solve(
        self,
        problem: KeplerProblem,
        E0: float,
        tol: float = 1e-14,
        max_iter: int = 50,
        record_history: bool = False,
        E_reference: float | None = None,
    ) -> SolveResult:
        """Run the iteration. Set ``tol=0.0`` to force exactly ``max_iter``
        iterations, which is what the convergence-order study needs."""
        problem.reset_cost()
        state = self.init_state(problem, E0)

        E = E0
        history: list[IterationRecord] = []
        failure: str | None = None
        converged = False
        n_iter = 0

        # Iteration 0 = the starting guess, so plots can show how much of the
        # win comes from the guess layer alone.
        residual = abs(problem.f(E0))
        if record_history:
            history.append(
                IterationRecord(
                    iteration=0,
                    E=E0,
                    residual=residual,
                    error=None if E_reference is None else abs(E0 - E_reference),
                    step=None,
                    cost=problem.cost.snapshot(),
                )
            )

        t0 = time.perf_counter()
        for n in range(1, max_iter + 1):
            try:
                E_new = self.step(problem, E, state)
            except (OverflowError, ZeroDivisionError, ValueError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break

            if not math.isfinite(E_new):
                failure = "non-finite iterate"
                break

            step_size = abs(E_new - E)
            E = E_new
            n_iter = n
            residual = abs(problem.f(E))

            if record_history:
                history.append(
                    IterationRecord(
                        iteration=n,
                        E=E,
                        residual=residual,
                        error=None if E_reference is None else abs(E - E_reference),
                        step=step_size,
                        cost=problem.cost.snapshot(),
                    )
                )

            if self._converged(residual, step_size, tol):
                converged = True
                break
        wall_time = time.perf_counter() - t0

        return SolveResult(
            solver=self.name,
            guess="",  # filled in by the runner, which knows the guess used
            e=problem.e,
            M=problem.M,
            E=E,
            converged=converged,
            iterations=n_iter,
            residual=residual,
            error=None if E_reference is None else abs(E - E_reference),
            history=history,
            cost=problem.cost.snapshot(),
            wall_time=wall_time,
            failure=failure,
        )
