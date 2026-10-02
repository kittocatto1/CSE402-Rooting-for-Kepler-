"""Safeguarded NWM9 and NWM11, plus their memoryless controls.

Why this file exists
--------------------
The grid benchmark shows that NWM9 and NWM11 fail far more often than Danby
in the hard corner (e > 0.9, M < 0.1). Switching the memory off entirely
(every self-accelerating parameter set to zero) removes most of those
failures, so the failures come from the memory, not from the eighth-order
base schemes. Switching it off everywhere, though, also throws away the
order lift the papers prove.

The safeguard keeps the memory where it is safe and drops it where it is
not. Each parameter enters a denominator as a correction term added to a
derivative estimate:

    NWM9   third step:  ... + alpha_n f(z_n)      added to ~f'(z_n)
    NWM11  first step:  f'(s_k) + alpha_k f(s_k)
    NWM11  third step:  w'      + beta_k  f(t_k)

The papers derive these parameters assuming the iterate is already close to
the root, where f is tiny and the correction term is a small nudge. Far from
the root, or where f' is nearly zero, the term can be as large as the
derivative it is supposed to correct, and the step is thrown far away. The
rule is therefore:

    use the parameter only if |parameter * f| <= GUARD * |f'|

otherwise use zero for that iteration (the memoryless base step). Near a
simple root f -> 0 while f' stays finite, so the test always passes there
and the asymptotic order is exactly the paper's. The tests check this at
2000 digits.

f'(s_k) is the scale in every test, including NWM11's third step: it is the
one derivative actually evaluated in the iteration, and w' and the NWM9
third-step bracket are estimates of the same derivative at nearby points.

Owner: Mahdi.
"""

from __future__ import annotations

from keplerbench.core.registry import register_solver
from keplerbench.solvers.nwm9 import NWM9Solver
from keplerbench.solvers.nwm11 import NWM11Solver

#: Largest allowed size of the correction term, as a fraction of |f'|.
#: With 1/2, a guarded denominator f' + param*f stays between 1/2 and 3/2
#: times f' with the same sign, so the parameter can at most double the
#: Newton-like step and never reverse it. Not tuned; it is the only value
#: the committed results use.
GUARD = 0.5


def _small_enough(param, f_value, fprime) -> bool:
    return abs(param * f_value) <= GUARD * abs(fprime)


@register_solver("nwm9_guarded")
class NWM9GuardedSolver(NWM9Solver):
    """NWM9 that drops alpha_n for any step where it is not a small correction."""

    category = "variant"

    def _accelerating_parameter(self, memory, x, y, z, fx, fy, fz, fpx, problem):
        alpha = super()._accelerating_parameter(memory, x, y, z, fx, fy, fz, fpx, problem)
        return alpha if _small_enough(alpha, fz, fpx) else 0 * alpha


@register_solver("nwm11_guarded")
class NWM11GuardedSolver(NWM11Solver):
    """NWM11 that drops alpha_k or beta_k for any step where it is not a small correction."""

    category = "variant"

    def _alpha(self, memory, s, fs, fps, problem):
        alpha, alpha_term = super()._alpha(memory, s, fs, fps, problem)
        if _small_enough(alpha, fs, fps):
            return alpha, alpha_term
        # Returning None for the term also tells _beta there is no usable
        # alpha to reuse this iteration, so beta falls back to its stored value.
        return 0 * alpha, None

    def _beta(self, memory, s, v, t, fs, fv, ft, fps, alpha_term, problem):
        beta = super()._beta(memory, s, v, t, fs, fv, ft, fps, alpha_term, problem)
        return beta if _small_enough(beta, ft, fps) else 0 * beta


@register_solver("nwm9_memoryless")
class NWM9MemorylessSolver(NWM9Solver):
    """Control: alpha_n fixed at zero. This is the eighth-order base method."""

    category = "variant"

    theoretical_order = 8.0

    def _accelerating_parameter(self, *args, **kwargs):
        return 0.0


@register_solver("nwm11_memoryless")
class NWM11MemorylessSolver(NWM11Solver):
    """Control: alpha_k = beta_k = 0. This is the eighth-order base method."""

    category = "variant"

    theoretical_order = 8.0

    def _alpha(self, *args, **kwargs):
        return 0.0, None

    def _beta(self, *args, **kwargs):
        return 0.0
