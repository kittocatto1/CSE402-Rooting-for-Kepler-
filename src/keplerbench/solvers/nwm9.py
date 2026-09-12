"""NWM9 - optimal 8th-order base + one memory parameter.

Reference
---------
S. K. Mittal, S. Panday and L. Jantschi, "Enhanced Ninth-Order Memory-Based
Iterative Technique for Efficiently Solving Nonlinear Equations",
Mathematics 12(22), 3490, 2024.  doi:10.3390/math12223490

Claimed order: 8.8989 (the optimal 8th-order without-memory scheme of
Matthies, Salimi, Sharifi & Varona (2016) lifted by a single self-accelerating
parameter placed in the THIRD step).  Efficiency index 1.6818 -> 1.7272.

NOTE: the proposal describes the with-memory methods as "derivative-free".
They are not - the first step is a plain Newton step and needs f'(x_n).  For
Kepler that is good news rather than bad: f and f' come from ONE sincos pair,
which is exactly why the cost claim below works out.

Role in the benchmark: the controlled, lower-order sibling of NWM11.  Having
both lets us say whether any advantage comes from "with-memory" as an idea or
from the specific NWM11 construction.  (Strictly, NWM11's own same-paper
sibling is NWM10; NWM9 is a different construction from a different paper.
See the note in nwm11.py.)

Cost per iteration (Section 4.1 accounting):
  * distinct evaluation points : 3   (x_n, y_n, z_n)
  * full (sin, cos) pairs      : 1   (at x_n, for f and f' together)
  * sin-only                   : 2   (at y_n and z_n)
  * synthesised derivatives    : 3   (H5'', H6''', H7'''' - memory steps only)

Owner: Suchi.
"""

from __future__ import annotations

from typing import Any

from keplerbench.core.base import IterativeSolver
from keplerbench.core.registry import register_solver
from keplerbench.core.types import KeplerProblem
from keplerbench.solvers._withmemory_base import (
    WithMemoryMixin,
    hermite_derivative_estimate,
    newton_divided_differences,
)

#: Paper's initial value for the accelerating parameter, Section 3: "for
#: NWM9, we set the parameter value alpha0 to 0.01".  Only used on the first
#: iteration, where there is no history to interpolate through.
ALPHA_0 = 0.01


@register_solver("nwm9")
class NWM9Solver(WithMemoryMixin, IterativeSolver):
    """Three-step scheme with one accelerating parameter in the third step.

    Per iteration (paper's Eq. (2) with alpha -> alpha_n, Eq. (8)):

        y_n = x_n - f(x_n)/f'(x_n)

        z_n = x_n - (f(x_n)/f'(x_n)) * [ 1 + u + (1 + 1/(1 + t)) * u**2 ]
              with u = f(y_n)/f(x_n)  and  t = f(x_n)/f'(x_n)

        x_{n+1} = z_n - f(z_n) / ( f[z,y] + (z-y) f[z,y,x]
                                   + (z-y)(z-x) f[z,y,x,x] + alpha_n f(z_n) )

    The bracket in the z step was verified against the paper's own error
    expression, Eq. (6): e_z = c2 (c2 + 5 c2^2 - c3) e^4 + O(e^5).  The text
    layer of the PDF renders that nested fraction ambiguously and several
    readings give order 8, so order alone does not pin it down - the error
    CONSTANT does.  See tests/test_solvers_withmemory.py.

    Note that f[z,y,x,x] is a confluent divided difference: it needs f'(x_n),
    which the first step already paid for.
    """

    theoretical_order = 8.8989
    category = "with-memory"
    reference = "Mittal, Panday & Jantschi (2024), doi:10.3390/math12223490"

    def init_state(self, problem: KeplerProblem, E0: float) -> dict[str, Any]:
        state = self._fresh_state()
        state["memory"].params["alpha"] = ALPHA_0
        return state

    # ------------------------------------------------------------------
    def _accelerating_parameter(self, memory, x, y, z, fx, fy, fz, fpx, problem):
        """alpha_n from Eq. (9), or the stored value when memory is unusable.

        Eq. (9):

            alpha_n = - H7''''(z_n) f'(x_n)
                        / ( 12 H5''(x_n) f'(x_n) + 30 (H5''(x_n))**2
                            - 4 H6'''(y_n) f'(x_n) )
                      - H5''(x_n) / (2 f'(x_n))

        The three Hermite polynomials interpolate through this iteration's
        nodes and the previous one's, with x_n and x_{n-1} as double nodes -
        which is why the divided-difference helper takes derivative values.
        """
        if not memory.has_memory():
            return memory.params.get("alpha", ALPHA_0)

        x_p, y_p, z_p = memory.prev_points
        fx_p, fy_p, fz_p = memory.prev_values
        fpx_p = memory.prev_derivatives[0]
        if fpx_p is None:
            return memory.params.get("alpha", ALPHA_0)

        # Nodes exactly as printed in the paper.  Order matters: the repeated
        # nodes must stay adjacent, and the polynomial is differentiated at
        # its own leading node.
        h5_x = [x, x, z_p, y_p, x_p, x_p]
        h5_f = [fx, fx, fz_p, fy_p, fx_p, fx_p]
        h5_d = [fpx, None, None, None, fpx_p, None]

        h6_x = [y, x, x, z_p, y_p, x_p, x_p]
        h6_f = [fy, fx, fx, fz_p, fy_p, fx_p, fx_p]
        h6_d = [None, fpx, None, None, None, fpx_p, None]

        h7_x = [z, y, x, x, z_p, y_p, x_p, x_p]
        h7_f = [fz, fy, fx, fx, fz_p, fy_p, fx_p, fx_p]
        h7_d = [None, None, fpx, None, None, None, fpx_p, None]

        try:
            h5 = hermite_derivative_estimate(h5_x, h5_f, order=2, at=x, ds=h5_d)
            h6 = hermite_derivative_estimate(h6_x, h6_f, order=3, at=y, ds=h6_d)
            h7 = hermite_derivative_estimate(h7_x, h7_f, order=4, at=z, ds=h7_d)
        except (ZeroDivisionError, ValueError):
            # Nodes collapsed onto each other near convergence.  Keeping the
            # previous parameter costs a little order on the final step; a nan
            # would destroy the iterate outright.
            return memory.params.get("alpha", ALPHA_0)

        self._note_synthesised(problem, 3)

        denominator = 12 * h5 * fpx + 30 * h5 * h5 - 4 * h6 * fpx
        if denominator == 0 or fpx == 0:
            return memory.params.get("alpha", ALPHA_0)

        alpha = -(h7 * fpx) / denominator - h5 / (2 * fpx)
        if alpha != alpha or alpha in (float("inf"), float("-inf")):
            return memory.params.get("alpha", ALPHA_0)
        return alpha

    # ------------------------------------------------------------------
    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        memory = state["memory"]
        x = E

        # Step 1 - Newton. One sincos pair buys both f and f'.
        fx, fpx = problem.f_fprime(x)
        if fpx == 0:
            raise ZeroDivisionError("f'(x) = 0; the Newton sub-step is undefined")
        t = fx / fpx
        y = x - t

        # Step 2 - the order-4 correction. f only, so sin alone.
        fy = problem.f(y)
        if fx == 0:
            return x  # already exactly on the root
        u = fy / fx
        if (1 + t) == 0:
            raise ZeroDivisionError("1 + f(x)/f'(x) = 0 in the z sub-step")
        z = x - t * (1 + u + (1 + 1 / (1 + t)) * u * u)

        # Step 3 - the accelerated finish. f only again.
        fz = problem.f(z)

        # Near convergence the three nodes collapse: once the order-4
        # correction is below one ulp of x, z == y (or z == x) exactly and the
        # difference quotients f[z,y], f[z,y,x] are 0/0.  That is not a
        # failure - it means z already sits at the precision floor, so it is
        # the best answer this iteration can produce.  Returning it lets the
        # harness's own stopping test see the converged residual, instead of
        # the solver dying one step short of the root.
        if z == y or z == x:
            return z

        alpha = self._accelerating_parameter(
            memory, x, y, z, fx, fy, fz, fpx, problem
        )

        # f[z,y], f[z,y,x] and f[z,y,x,x] are the leading coefficients of the
        # confluent table over [z, y, x, x] - one call gives all three.
        coefficients = newton_divided_differences(
            [z, y, x, x], [fz, fy, fx, fx], [None, None, fpx, None]
        )
        denominator = (
            coefficients[1]
            + (z - y) * coefficients[2]
            + (z - y) * (z - x) * coefficients[3]
            + alpha * fz
        )
        if denominator == 0:
            raise ZeroDivisionError("third-step denominator vanished")

        x_next = z - fz / denominator

        memory.params["alpha"] = alpha
        memory.record_iteration(
            points=[x, y, z],
            values=[fx, fy, fz],
            derivatives=[fpx, None, None],
        )
        return x_next
