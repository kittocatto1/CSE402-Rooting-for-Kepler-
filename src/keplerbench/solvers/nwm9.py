"""NWM9 - optimal 8th-order derivative-free base + one memory parameter.

Reference
---------
R. Mittal, A. Panday and L. Jantschi, "An Optimal Higher-Order Derivative-Free
with-Memory Iterative Scheme for Solving Nonlinear Equations", Symmetry 16(5),
588, 2024.  doi:10.3390/sym16050588

Claimed order: 8.8989 (an 8th-order memoryless base lifted by a single
self-accelerating parameter).

Role in the benchmark: the controlled, lower-order sibling of NWM11.  Having
both lets us say whether any advantage comes from "with-memory" as an idea or
from the specific NWM11 construction.

Cost per iteration (fill in from the paper, Section 4.1):
  * distinct evaluation points : ?
  * full (sin, cos) pairs      : ?
  * sin-only / cos-only        : ?
  * synthesised derivatives    : ?  (all the Hermite ones)

Owner: Suchi.
"""

from __future__ import annotations

from typing import Any

from keplerbench.core.base import IterativeSolver
from keplerbench.core.registry import register_solver
from keplerbench.core.types import KeplerProblem
from keplerbench.solvers._withmemory_base import WithMemoryMixin


@register_solver("nwm9")
class NWM9Solver(WithMemoryMixin, IterativeSolver):
    """Three-step derivative-free scheme with one accelerating parameter."""

    theoretical_order = 8.8989
    category = "with-memory"
    reference = "Mittal, Panday & Jantschi (2024), doi:10.3390/sym16050588"

    def init_state(self, problem: KeplerProblem, E0: float) -> dict[str, Any]:
        return self._fresh_state()

    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        # TODO(Suchi): implement the scheme. Suggested order of work:
        #
        #   A. Transcribe the MEMORYLESS base method first (the optimal
        #      8th-order derivative-free scheme in the paper) and verify it
        #      reaches order ~8 on the paper's own test functions. Do this
        #      BEFORE adding memory - otherwise you cannot tell which half is
        #      broken.
        #
        #   B. Add the self-accelerating parameter:
        #        - on the first iteration there is no history, so use the
        #          paper's stated initial value for the parameter
        #          (state["memory"].first_iteration is True there);
        #        - afterwards recompute it each iteration from the Hermite
        #          interpolation helpers in _withmemory_base.py.
        #
        #   C. Store what the next iteration needs in state["memory"]
        #      (prev_points, prev_values, params) and set first_iteration
        #      to False.
        #
        #   D. Call self._note_synthesised(problem, n) for every derivative
        #      you obtained by interpolation instead of evaluation - the
        #      cost table in the report depends on this being honest.
        #
        #   E. Kepler-specific: wherever the scheme needs f at a new point,
        #      check whether you also need cos there. If yes use
        #      problem.f_fprime (one sincos pair); if you only need f use
        #      problem.f (sin only). Picking the right call is exactly what
        #      Section 4.1 is measuring - do not default to f_fprime.
        #
        # Verification target (Work Plan step 2): empirical order from
        # evaluation/convergence_order.py should land near 8.8989 on the
        # paper's test set before this is used on the (e, M) grid.
        raise NotImplementedError("NWM9Solver.step: see TODO above")
