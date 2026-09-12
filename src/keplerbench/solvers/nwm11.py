"""NWM11 - bi-parametric with-memory scheme, the project's primary "new" method.

Reference
---------
S. K. Mittal, S. Panday, L. Jantschi and L. C. Bolundut, "Two novel efficient
memory-based multi-point iterative methods for solving nonlinear equations",
AIMS Mathematics 10(3), 5421-5443, 2025.  doi:10.3934/math.2025250

Claimed order: 10.7446, from TWO self-accelerating parameters placed in the
first and third steps of the optimal eighth-order without-memory method of
Solaiman & Hashim.  Efficiency index 1.6818 -> 1.8105.

The same paper defines NWM10 (one parameter, R-order 10) as the uni-parametric
sibling.  Note for the report: the proposal calls NWM9 the "controlled,
lower-order sibling" of NWM11, but NWM9 is a different construction from a
different paper (Mathematics 12(22), 3490).  NWM10 is the true same-base
control.  Worth stating explicitly rather than letting the table imply
otherwise.

Per the proposal's method table this uses 3 evaluation points but only ONE
full (sin, cos) pair per iteration - which, if true, is precisely the reason
it might beat Danby on Kepler despite doing more work on paper.  Confirmed
with the cost counters; see tests.

Cost per iteration (Section 4.1 accounting):
  * distinct evaluation points : 3   (s_k, v_k, t_k)
  * full (sin, cos) pairs      : 1   (at s_k, for f and f' together)
  * sin-only                   : 2   (at v_k and t_k)
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
)

#: Paper's initial parameter values, Section 3.
ALPHA_0 = 0.01
BETA_0 = 0.00001


@register_solver("nwm11")
class NWM11Solver(WithMemoryMixin, IterativeSolver):
    """Multipoint scheme with two accelerating parameters, Eq. (2.33).

        v_k     = s_k - f(s_k) / (f'(s_k) + alpha_k f(s_k))

        t_k     = v_k - f(v_k)/q - 2 f(v_k)^2 q R
                              / (4 q^4 - 4 f(v_k) q^2 R + f(v_k)^2 R^2)

        s_{k+1} = t_k - f(t_k) / (w' + beta_k f(t_k))

    with the auxiliary approximations of the base method

        q  ~ f'(v_k)     = 2 f[v_k, s_k] - f'(s_k)
        R  ~ f''(v_k)    = 2 (f'(s_k) - f[v_k, s_k]) / (s_k - v_k)
        w' ~ f'(t_k)     = f[t_k,s_k] (2 + (s_k-t_k)/(v_k-t_k))
                           - (s_k-t_k)^2 / ((s_k-v_k)(v_k-t_k)) f[v_k,s_k]
                           + f'(s_k) (v_k-t_k)/(s_k-v_k)

    The ``w'`` grouping and the two-form definition of ``R`` are ambiguous in
    the PDF's text layer, so both were pinned against the paper's own error
    expressions (2.4)-(2.6) rather than read off by eye - see
    tests/test_solvers_withmemory.py.
    """

    theoretical_order = 10.7446
    category = "with-memory"
    reference = "Mittal, Panday, Jantschi & Bolundut (2025), doi:10.3934/math.2025250"

    def init_state(self, problem: KeplerProblem, E0: float) -> dict[str, Any]:
        state = self._fresh_state()
        state["memory"].params["alpha"] = ALPHA_0
        state["memory"].params["beta"] = BETA_0
        return state

    # ------------------------------------------------------------------
    def _hermite_nodes(self, memory, s, v, t, fs, fv, ft, fps):
        """The three node lists of Eqs (2.9) and (2.34), with their values.

        s_k and s_{k-1} appear twice in every one of them, which is what
        makes these Hermite rather than plain Newton interpolants.
        """
        s_p, v_p, t_p = memory.prev_points
        fs_p, fv_p, ft_p = memory.prev_values
        fps_p = memory.prev_derivatives[0]

        h5 = ([s, s, t_p, v_p, s_p, s_p],
              [fs, fs, ft_p, fv_p, fs_p, fs_p],
              [fps, None, None, None, fps_p, None])
        h6 = ([v, s, s, t_p, v_p, s_p, s_p],
              [fv, fs, fs, ft_p, fv_p, fs_p, fs_p],
              [None, fps, None, None, None, fps_p, None])
        h7 = ([t, v, s, s, t_p, v_p, s_p, s_p],
              [ft, fv, fs, fs, ft_p, fv_p, fs_p, fs_p],
              [None, None, fps, None, None, None, fps_p, None])
        return h5, h6, h7

    def _alpha(self, memory, s, fs, fps, problem):
        """alpha_k = -H5''(s_k) / (2 f'(s_k)),  Eq. (2.8).

        Computed at the TOP of the iteration: H5 needs only s_k and the
        previous iteration's nodes, all of which are already in hand.
        """
        if not memory.has_memory() or memory.prev_derivatives[0] is None:
            return memory.params.get("alpha", ALPHA_0), None

        s_p, v_p, t_p = memory.prev_points
        fs_p, fv_p, ft_p = memory.prev_values
        fps_p = memory.prev_derivatives[0]
        try:
            h5 = hermite_derivative_estimate(
                [s, s, t_p, v_p, s_p, s_p],
                [fs, fs, ft_p, fv_p, fs_p, fs_p],
                order=2, at=s,
                ds=[fps, None, None, None, fps_p, None],
            )
        except (ZeroDivisionError, ValueError):
            return memory.params.get("alpha", ALPHA_0), None

        self._note_synthesised(problem, 1)
        if fps == 0:
            return memory.params.get("alpha", ALPHA_0), None
        alpha = -h5 / (2 * fps)
        if not _finite(alpha):
            return memory.params.get("alpha", ALPHA_0), None
        return alpha, alpha

    def _beta(self, memory, s, v, t, fs, fv, ft, fps, alpha_term, problem):
        """beta_k = H7''''(t_k) / (4 H6'''(v_k)) - H5''(s_k)/(2 f'(s_k)),
        Eq. (2.34).

        The trailing term is exactly alpha_k, so it is reused rather than
        recomputed - one interpolation saved per iteration.
        """
        if not memory.has_memory() or alpha_term is None:
            return memory.params.get("beta", BETA_0)

        _, h6_spec, h7_spec = self._hermite_nodes(memory, s, v, t, fs, fv, ft, fps)
        try:
            h6 = hermite_derivative_estimate(*h6_spec[:2], order=3, at=v, ds=h6_spec[2])
            h7 = hermite_derivative_estimate(*h7_spec[:2], order=4, at=t, ds=h7_spec[2])
        except (ZeroDivisionError, ValueError):
            return memory.params.get("beta", BETA_0)

        self._note_synthesised(problem, 2)
        if h6 == 0:
            return memory.params.get("beta", BETA_0)
        beta = h7 / (4 * h6) + alpha_term
        if not _finite(beta):
            return memory.params.get("beta", BETA_0)
        return beta

    # ------------------------------------------------------------------
    def step(self, problem: KeplerProblem, E: float, state: dict[str, Any]) -> float:
        memory = state["memory"]
        s = E

        # One sincos pair buys f and f' at s_k.
        fs, fps = problem.f_fprime(s)
        if fs == 0:
            return s

        alpha, alpha_term = self._alpha(memory, s, fs, fps, problem)

        # Step 1 - accelerated Newton.
        denominator = fps + alpha * fs
        if denominator == 0:
            raise ZeroDivisionError("f'(s) + alpha f(s) = 0 in the first sub-step")
        v = s - fs / denominator
        if v == s:
            return v

        # Step 2 - the order-8 core. f only at v_k, so sin alone.
        fv = problem.f(v)
        divided_vs = (fv - fs) / (v - s)
        q = 2 * divided_vs - fps
        R = 2 * (fps - divided_vs) / (s - v)
        if q == 0:
            raise ZeroDivisionError("q(v) = 0; the second sub-step is undefined")
        core = 4 * q**4 - 4 * fv * q**2 * R + fv**2 * R**2
        if core == 0:
            raise ZeroDivisionError("second-step denominator vanished")
        t = v - fv / q - (2 * fv**2 * q * R) / core
        if t == v or t == s:
            return t

        # Step 3 - accelerated finish. f only at t_k.
        ft = problem.f(t)
        beta = self._beta(memory, s, v, t, fs, fv, ft, fps, alpha_term, problem)

        divided_ts = (ft - fs) / (t - s)
        w = (
            divided_ts * (2 + (s - t) / (v - t))
            - (s - t) ** 2 / ((s - v) * (v - t)) * divided_vs
            + fps * (v - t) / (s - v)
        )
        final = w + beta * ft
        if final == 0:
            raise ZeroDivisionError("third-step denominator vanished")
        s_next = t - ft / final

        memory.params["alpha"] = alpha
        memory.params["beta"] = beta
        memory.record_iteration(
            points=[s, v, t],
            values=[fs, fv, ft],
            derivatives=[fps, None, None],
        )
        return s_next


def _finite(x) -> bool:
    """True when x is neither nan nor infinite, for float or mpf alike."""
    return x == x and x not in (float("inf"), float("-inf"))
