"""Danby's quartic method - RadVel's production solver.

Reference
---------
J. M. A. Danby, "The solution of Kepler's equation. I", Celestial Mechanics
40(3-4), 303-312, 1987.  doi:10.1007/BF01230252

Why it is in the benchmark: it is the solver RadVel actually ships, so it is
the "current practical standard" the newer methods have to beat, not a
strawman.

Cost per iteration (fill in once implemented, Section 4.1):
  * distinct evaluation points : 1 (expected)
  * full (sin, cos) pairs      : 1 (expected - f, f', f'', f''' all come from
                                    one sincos call, which is the point)
  * synthesised derivatives    : 0

Owner: Dipit.
"""

from __future__ import annotations

from typing import Any

from keplerbench.core.base import IterativeSolver
from keplerbench.core.registry import register_solver
from keplerbench.core.types import KeplerProblem


@register_solver("danby")
class DanbySolver(IterativeSolver):
    """Fourth-order single-point scheme using f, f', f'' and f'''.

    Shape of the update (VERIFY every line against Danby 1987 before
    trusting it - do not take this docstring as authoritative):

        f0 = E - e sin E - M
        f1 = 1 - e cos E          (= f')
        f2 = e sin E              (= f'')
        f3 = e cos E              (= f''')

        d1 = -f0 / f1
        d2 = -f0 / (f1 + d1 * f2 / 2)
        d3 = -f0 / (f1 + d2 * f2 / 2 + d2**2 * f3 / 6)
        E_next = E + d3

    Note how f0..f3 all come out of ONE sin/cos evaluation - use
    ``problem.derivatives(E, order=3)`` so the counter records exactly one
    sincos pair.
    """

    theoretical_order = 4.0
    category = "established"
    reference = "Danby (1987), doi:10.1007/BF01230252"

    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        # TODO(Dipit):
        #   1. f0, f1, f2, f3 = problem.derivatives(E, order=3)
        #   2. Implement the three nested corrections d1, d2, d3.
        #   3. Raise ZeroDivisionError if any denominator is 0 - the base
        #      class turns that into a recorded failure instead of a crash,
        #      which is what the robustness metric needs.
        #   4. Cross-check against radvel.kepler on ~1000 random (e, M):
        #      the roots must agree to ~1e-14.
        raise NotImplementedError("DanbySolver.step: see TODO above")
