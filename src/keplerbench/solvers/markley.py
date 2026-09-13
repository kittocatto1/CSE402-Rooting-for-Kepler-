"""Markley's near-analytic Kepler solver.

Reference
---------
F. L. Markley, "Kepler's Equation Solver", Celestial Mechanics and Dynamical
Astronomy 63(1), 101-111, 1995.  doi:10.1007/BF00691917

This one is structurally different from the rest: it is NOT an iteration.  It
builds a starting value from a cubic/quartic in closed form and then applies a
single fixed correction, so it always costs the same.  That is why it
subclasses ``KeplerSolver`` directly instead of ``IterativeSolver``, and why
its "order" column in the methods table is a dash.

Because it is closed-form, the guess layer does NOT apply to it in the same
way - see README, "How Markley is treated in the grid".

Cost (fill in once implemented, Section 4.1):
  * distinct evaluation points : ?
  * full (sin, cos) pairs      : ?
  * synthesised derivatives    : ?

Owner: Dipit.
"""

from __future__ import annotations

import math
import time

from keplerbench.core.base import KeplerSolver
from keplerbench.core.registry import register_solver
from keplerbench.core.types import KeplerProblem, SolveResult


@register_solver("markley")
class MarkleySolver(KeplerSolver):
    """Closed-form starter + one bounded correction.

    Stages in the paper (transcribe each one, do not guess):
      1. Reduce M to [0, pi] using the symmetry of Kepler's equation.
      2. Build the auxiliary quantities alpha, d, q, r, w from (e, M).
      3. Form the closed-form estimate E1 from those.
      4. Apply the final high-order correction using f, f', f'', f''' at E1.
      5. Undo the reduction from step 1.
    """

    theoretical_order = None  # closed form - no asymptotic order to report
    category = "established"
    reference = "Markley (1995), doi:10.1007/BF00691917"

    # ------------------------------------------------------------------
    # Stage 1: range reduction (Kepler's equation is odd about M = 0 and
    # 2*pi-periodic in M).  We fold M into [0, pi], solve there, and undo
    # the fold afterwards.  Everything is returned in a form that lets the
    # correction run in the ORIGINAL frame, so problem.f (which uses the
    # original M) stays consistent.
    # ------------------------------------------------------------------
    def _reduce_M(self, M: float):
        tau = 2.0 * math.pi
        revs = math.floor(M / tau)
        M0 = M - revs * tau  # now in [0, 2*pi)
        if M0 > math.pi:
            return tau - M0, revs, True
        return M0, revs, False 
    
    def _closed_form_estimate(self, e: float, Mr: float):
        
        pi = math.pi
        alpha = (3.0 * pi * pi + 1.6 * pi * (pi - Mr) / (1.0 + e)) / (pi * pi - 6.0)
        d = 3.0 * (1.0 - e) + alpha * e
        q = 2.0 * alpha * d * (1.0 - e) - Mr * Mr
        r = 3.0 * alpha * d * (d - 1.0 + e) * Mr + Mr ** 3
        w = (abs(r) + math.sqrt(q ** 3 + r * r)) ** (2.0 / 3.0)
        return (2.0 * r * w / (w * w + w * q + q * q) + Mr) / d

    def _refine(self, problem: KeplerProblem, E_est: float) -> float:
        """Markley's own fifth-order refinement, applied exactly once
        (Markley 1995, Eqs. 21-29).  All of f, f', f'', f''', f'''' come from a
        single sincos pair, so the whole method costs one transcendental call.

            f0 = E - e sinE - M   f1 = 1 - e cosE
            f2 = e sinE (= f'')    f3 = e cosE (= f''')    f4 = -e sinE (= -f2)

            d3 = -f0 / (f1 - f0 f2 / (2 f1))                      # Eq. 22 (Halley)
            d4 = -f0 / (f1 + d3 f2 / 2 + d3^2 f3 / 6)             # Eq. 23
            d5 = -f0 / (f1 + d4 f2 / 2 + d4^2 f3 / 6 + d4^3 f4/24)# Eq. 24
            E  = E_est + d5                                        # Eq. 29
        """
        f0, f1, f2, f3 = problem.derivatives(E_est, order=3)
        f4 = -f2  
        d3 = -f0 / (f1 - 0.5 * f0 * f2 / f1)
        d4 = -f0 / (f1 + 0.5 * d3 * f2 + d3 ** 2 * f3 / 6.0)
        d5 = -f0 / (f1 + 0.5 * d4 * f2 + d4 ** 2 * f3 / 6.0 + d4 ** 3 * f4 / 24.0)
        return E_est + d5

    def solve(
        self,
        problem: KeplerProblem,
        E0: float,
        tol: float = 1e-14,
        max_iter: int = 50,
        record_history: bool = False,
        E_reference: float | None = None,
    ) -> SolveResult:
        """Ignores ``E0``, ``tol`` and ``max_iter`` - it is not an iteration.

        Keep the signature identical anyway so the runner can treat every
        solver the same. Record ``iterations = 0``.
        """
        problem.reset_cost()
        t0 = time.perf_counter()

        # Stage 1: fold M into [0, pi].
        Mr, revs, flip = self._reduce_M(problem.M)
        # Stages 2-3: closed-form estimate in the reduced frame.
        E_est = self._closed_form_estimate(problem.e, Mr)
        # Stage 5 (undo the fold) done here so the correction below runs in the
        # original frame, where problem.f uses the original M.
        tau = 2.0 * math.pi
        if flip:
            E_est = tau - E_est
        E_est += revs * tau
        # Stage 4: single high-order correction in the original frame.
        E = self._refine(problem, E_est)

        wall_time = time.perf_counter() - t0
        return SolveResult(
            solver=self.name, guess="", e=problem.e, M=problem.M,
            E=E, converged=True, iterations=0,
            residual=abs(problem.f(E)),
            error=None if E_reference is None else abs(E - E_reference),
            history=[], cost=problem.cost.snapshot(),
            wall_time=wall_time, failure=None,
        )
