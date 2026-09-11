"""NWM11 - bi-parametric with-memory scheme, the project's primary "new" method.

Reference
---------
R. Mittal, A. Panday, L. Jantschi and L.-M. Bolundut, "High-Efficiency
With-Memory Iterative Solvers for Nonlinear Equations: Development,
Theoretical Analysis, and Numerical Validation", Mathematics 13(2), 245, 2025.
doi:10.3390/math13020245

Claimed order: 10.7446, from TWO self-accelerating parameters.

Per the proposal's method table this uses 3 evaluation points but only ONE
full (sin, cos) pair per iteration - which, if true, is precisely the reason
it might beat Danby on Kepler despite doing more work on paper.  Confirm that
claim with the cost counters rather than assuming it.

Owner: Suchi.
"""

from __future__ import annotations

from typing import Any

from keplerbench.core.base import IterativeSolver
from keplerbench.core.registry import register_solver
from keplerbench.core.types import KeplerProblem
from keplerbench.solvers._withmemory_base import WithMemoryMixin


@register_solver("nwm11")
class NWM11Solver(WithMemoryMixin, IterativeSolver):
    """Multipoint derivative-free scheme with two accelerating parameters."""

    theoretical_order = 10.7446
    category = "with-memory"
    reference = "Mittal, Panday, Jantschi & Bolundut (2025), doi:10.3390/math13020245"

    def init_state(self, problem: KeplerProblem, E0: float) -> dict[str, Any]:
        return self._fresh_state()

    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        # TODO(Suchi): same staged approach as NWM9 (A-E in nwm9.py), with
        # two differences:
        #
        #   * There are TWO parameters, and the paper computes them from
        #     interpolating polynomials of DIFFERENT degrees. Keep them in
        #     state["memory"].params under clear names and update both every
        #     iteration.
        #
        #   * The proposal claims 3 evaluation points / 1 sincos pair per
        #     iteration. After implementing, run one solve with
        #     record_history=True and read problem.cost.snapshot() to check
        #     the counts match. If they do not, either the implementation or
        #     the claim in our methods table is wrong - resolve it before the
        #     grid run, and report the measured numbers in the report.
        #
        # Verification target (Work Plan step 2): empirical order near
        # 10.7446 on the paper's test set.
        raise NotImplementedError("NWM11Solver.step: see TODO above")
