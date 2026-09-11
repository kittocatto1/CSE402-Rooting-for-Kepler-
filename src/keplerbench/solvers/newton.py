"""Newton-Raphson - the textbook anchor (Proposal Section 3.2, row 1).

WORKED EXAMPLE - fully implemented so the team has a reference for the
interface, the cost accounting and the docstring style.  Everything else in
this folder is a skeleton.

Cost per iteration (Section 4.1 accounting):
  * 1 distinct evaluation point
  * 1 full (sin, cos) pair at that point
  * 0 synthesised derivatives

Owner: Dipit.
"""

from __future__ import annotations

from typing import Any

from keplerbench.core.base import IterativeSolver
from keplerbench.core.registry import register_solver
from keplerbench.core.types import KeplerProblem


@register_solver("newton")
class NewtonSolver(IterativeSolver):
    """E_{n+1} = E_n - f(E_n) / f'(E_n).

    With f(E) = E - e sin E - M and f'(E) = 1 - e cos E, one sincos call at
    E_n gives both, which is exactly the Kepler cost structure the proposal
    wants credited.
    """

    theoretical_order = 2.0
    category = "baseline"
    reference = "textbook"

    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        f, fprime = problem.f_fprime(E)
        if fprime == 0.0:
            raise ZeroDivisionError("f'(E) = 0; Newton step undefined")
        return E - f / fprime
