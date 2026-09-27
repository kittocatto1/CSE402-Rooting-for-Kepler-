"""NWM9 - an 8th-order method lifted to 8.8989 by one memory parameter.

Reference
---------
S. K. Mittal, S. Panday and L. Jantschi, "Enhanced Ninth-Order Memory-Based
Iterative Technique for Efficiently Solving Nonlinear Equations",
Mathematics 12(22), 3490, 2024.  doi:10.3390/math12223490

The paper takes the 8th-order method of Matthies et al. (2016) and adds one
tuning parameter to its third step. Claimed order 8.8989, efficiency index
1.6818 -> 1.7272.

NOTE: the proposal calls these methods "derivative-free". They are not. The
first step is an ordinary Newton step and needs f'(x). On Kepler that helps
rather than hurts, because f and f' come out of one sincos call - which is
why the cost numbers below work out.

Why it is in the benchmark: NWM9 was meant as the lower-order control for
NWM11. It is not really a matched pair, though - the two come from different
papers with different base methods. NWM11's true control is NWM10. See the
note in nwm11.py.

Cost per iteration:
  * evaluation points   : 3   (x, y, z)
  * (sin, cos) pairs    : 1   (at x, giving f and f' together)
  * sin only            : 2   (at y and z)
  * interpolated derivs : 3   (only once memory exists)

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

#: Starting value for the parameter, from the paper: "for NWM9, we set the
#: parameter value alpha0 to 0.01". Used only on the first iteration, when
#: there is no history to interpolate through yet.
ALPHA_0 = 0.01


@register_solver("nwm9")
class NWM9Solver(WithMemoryMixin, IterativeSolver):
    """Three steps per iteration, with the tuning parameter in the third.

    From the paper's Eq. (2), with alpha replaced by alpha_n (Eq. 8):

        y_n = x_n - f(x_n)/f'(x_n)

        z_n = x_n - (f(x_n)/f'(x_n)) * [ 1 + u + (1 + 1/(1 + t)) * u**2 ]
              with u = f(y_n)/f(x_n)  and  t = f(x_n)/f'(x_n)

        x_{n+1} = z_n - f(z_n) / ( f[z,y] + (z-y) f[z,y,x]
                                   + (z-y)(z-x) f[z,y,x,x] + alpha_n f(z_n) )

    The bracket in the z step deserves a warning. The PDF prints that nested
    fraction in a way that can be read several ways, and more than one of the
    readings still converges at order 8 - so checking the order does not tell
    you which is right. What does is the paper's error constant, Eq. (6):
    e_z = c2 (c2 + 5 c2^2 - c3) e^4. That check is in
    tests/test_solvers_withmemory.py.

    f[z,y,x,x] repeats x, so it needs f'(x) - which the first step has
    already computed, so it costs nothing.
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
        """Compute alpha_n, or reuse the last one if memory is not usable.

        The paper's Eq. (9):

            alpha_n = - H7''''(z_n) f'(x_n)
                        / ( 12 H5''(x_n) f'(x_n) + 30 (H5''(x_n))**2
                            - 4 H6'''(y_n) f'(x_n) )
                      - H5''(x_n) / (2 f'(x_n))

        The three polynomials are fitted through this iteration's nodes plus
        the previous iteration's. x appears twice in each node list, which is
        why the helper needs derivative values.
        """
        if not memory.has_memory():
            return memory.params.get("alpha", ALPHA_0)

        x_p, y_p, z_p = memory.prev_points
        fx_p, fy_p, fz_p = memory.prev_values
        fpx_p = memory.prev_derivatives[0]
        if fpx_p is None:
            return memory.params.get("alpha", ALPHA_0)

        # Node lists exactly as the paper prints them. Order matters: equal
        # nodes must stay next to each other, and each polynomial is
        # differentiated at its own first node.
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
            # Near convergence the nodes fall on top of each other. Reusing
            # the previous parameter loses a little order on the last step;
            # returning nan would wreck the iterate completely.
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

        # Step 1: a Newton step. One sincos call gives both f and f'.
        fx, fpx = problem.f_fprime(x)
        if fpx == 0:
            raise ZeroDivisionError("f'(x) = 0; the Newton sub-step is undefined")
        t = fx / fpx
        y = x - t

        # Step 2: the order-4 correction. Needs f only, so sin alone.
        fy = problem.f(y)
        if fx == 0:
            return x  # already exactly on the root
        u = fy / fx
        if (1 + t) == 0:
            raise ZeroDivisionError("1 + f(x)/f'(x) = 0 in the z sub-step")
        z = x - t * (1 + u + (1 + 1 / (1 + t)) * u * u)

        # Step 3: the accelerated finish. f only again.
        fz = problem.f(z)

        # Once the correction drops below one ulp of x, z equals y (or x)
        # exactly, and the quotients f[z,y] and f[z,y,x] become 0/0. That is
        # not an error: z already sits at the precision floor, so it is the
        # best answer this iteration can give. Return it and let the outer
        # loop see a converged residual, instead of failing one step short.
        if z == y or z == x:
            return z

        alpha = self._accelerating_parameter(
            memory, x, y, z, fx, fy, fz, fpx, problem
        )

        # f[z,y], f[z,y,x] and f[z,y,x,x] are the first coefficients of the
        # table over [z, y, x, x], so one call gives all three.
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
