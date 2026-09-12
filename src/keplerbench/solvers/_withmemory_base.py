"""Shared plumbing for the 2024-2025 with-memory schemes.

What "with memory" means here
-----------------------------
A normal (memoryless) method throws away everything it computed in iteration
n before starting iteration n+1.  A with-memory method keeps those values and
uses them to estimate one or more *self-accelerating parameters* by Hermite
interpolation.  Better parameters make the error constant smaller, which
raises the convergence order ABOVE the memoryless base method - and it costs
nothing extra, because the interpolation only reuses numbers already paid for.

That is the whole reason these methods are interesting for Kepler: extra
order without extra transcendental calls.  Whether that actually wins once
Kepler's sincos structure is costed is the question the project asks.

Why the interpolation needs derivative values
---------------------------------------------
The published parameter formulas interpolate over *repeated* nodes.  For the
bi-parametric scheme the nodes are

    H5 over [s_k, s_k, t_{k-1}, v_{k-1}, s_{k-1}, s_{k-1}]
    H6 over [v_k, s_k, s_k, t_{k-1}, v_{k-1}, s_{k-1}, s_{k-1}]
    H7 over [t_k, v_k, s_k, s_k, t_{k-1}, v_{k-1}, s_{k-1}, s_{k-1}]

A repeated node x forces the divided-difference table to use f'(x) where it
would otherwise divide by (x - x) = 0.  That is why the functions below take
an optional ``ds`` argument: a plain Newton table cannot express these
polynomials at all.  With ``ds=None`` and distinct nodes they reduce exactly
to the ordinary Newton divided differences.

Everything here is arithmetic-agnostic - floats in, floats out; mpmath.mpf
in, mpmath.mpf out - because these tables become badly conditioned as the
nodes collapse onto the root, and the verification step runs them at high
precision for exactly that reason.

Owner: Suchi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "MemoryState",
    "WithMemoryMixin",
    "newton_divided_differences",
    "hermite_derivative_estimate",
]


@dataclass
class MemoryState:
    """Values carried from one iteration to the next.

    The three "prev" lists are parallel and describe the *previous* full
    iteration: for the AIMS schemes they hold (s_{k-1}, v_{k-1}, t_{k-1}) and
    the function values there, plus f'(s_{k-1}) where it is known.
    """

    #: Iterates from the previous iteration (x_{n-1} and its sub-steps).
    prev_points: list[float] = field(default_factory=list)
    #: Function values at ``prev_points``, in the same order.
    prev_values: list[float] = field(default_factory=list)
    #: Derivative values at ``prev_points``, None where unknown.  Needed
    #: because the Hermite polynomials repeat s_{k-1} as a double node.
    prev_derivatives: list[float | None] = field(default_factory=list)
    #: Self-accelerating parameters. NWM9/NWM10 have one, NWM11 has two.
    params: dict[str, float] = field(default_factory=dict)
    #: True until the first full iteration has run - see note below.
    first_iteration: bool = True

    def has_memory(self) -> bool:
        """Whether enough history exists to compute the parameters."""
        return not self.first_iteration and len(self.prev_points) > 0

    def record_iteration(
        self,
        points: Sequence[float],
        values: Sequence[float],
        derivatives: Sequence[float | None] | None = None,
    ) -> None:
        """Overwrite the stored history with this iteration's nodes.

        Only one iteration of history is kept, which is all the published
        parameter formulas use.  Call this at the END of ``step``.
        """
        if len(points) != len(values):
            raise ValueError("points and values must be the same length")
        if derivatives is None:
            derivatives = [None] * len(points)
        elif len(derivatives) != len(points):
            raise ValueError("derivatives must match points in length")

        self.prev_points = list(points)
        self.prev_values = list(values)
        self.prev_derivatives = list(derivatives)
        self.first_iteration = False


# ----------------------------------------------------------------------
# Interpolation machinery
# ----------------------------------------------------------------------
def _check_repeats_are_grouped(xs: Sequence[float]) -> None:
    """Confluent divided differences require equal nodes to be adjacent.

    [a, a, b] is fine; [a, b, a] is not, and would silently produce a
    different (wrong) polynomial rather than an error.
    """
    for i in range(len(xs)):
        for k in range(i + 2, len(xs)):
            if xs[k] == xs[i] and xs[k - 1] != xs[i]:
                raise ValueError(
                    f"repeated node {xs[i]!r} at positions {i} and {k} must be "
                    "adjacent; reorder the nodes so equal ones are grouped"
                )


def newton_divided_differences(
    xs: Sequence[float],
    fs: Sequence[float],
    ds: Sequence[float | None] | None = None,
) -> list[float]:
    """Divided-difference table for Newton's interpolating polynomial.

    Returns the leading coefficients f[x0], f[x0,x1], f[x0,x1,x2], ...

    Both papers build their self-accelerating parameters out of exactly this,
    so it is factored out here and unit-tested once.

    Repeated nodes
    --------------
    When ``xs[i] == xs[i+1]`` the ordinary quotient would be 0/0; the
    confluent form uses the derivative instead, f[x, x] = f'(x).  Supply it
    through ``ds`` (a list parallel to ``xs``; entries for non-repeated nodes
    may be None).  Equal nodes must be adjacent.

    Only multiplicity 2 is supported, because only first derivatives are
    available - a node repeated three times would need f''.  That case raises
    rather than returning a quietly wrong table.
    """
    m = len(xs)
    if len(fs) != m:
        raise ValueError(f"xs and fs differ in length: {m} vs {len(fs)}")
    if ds is not None and len(ds) != m:
        raise ValueError(f"ds must have length {m}, got {len(ds)}")
    if m == 0:
        return []

    _check_repeats_are_grouped(xs)

    column = list(fs)
    coefficients = [column[0]]

    for order in range(1, m):
        nxt = []
        for i in range(m - order):
            denominator = xs[i + order] - xs[i]
            if denominator == 0:
                if order != 1:
                    raise ValueError(
                        f"node {xs[i]!r} repeats {order + 1} times; that needs "
                        f"a derivative of order {order}, but only first "
                        "derivatives are supported"
                    )
                if ds is None or ds[i] is None:
                    raise ValueError(
                        f"node {xs[i]!r} is repeated, so f'({xs[i]!r}) is "
                        "required - pass it in ds"
                    )
                nxt.append(ds[i])
            else:
                nxt.append((column[i + 1] - column[i]) / denominator)
        column = nxt
        coefficients.append(column[0])

    return coefficients


def hermite_derivative_estimate(
    xs: Sequence[float],
    fs: Sequence[float],
    order: int,
    at: float,
    ds: Sequence[float | None] | None = None,
) -> float:
    """Estimate the ``order``-th derivative of f at ``at`` by Hermite/Newton
    interpolation through the already-computed points.

    This is the machinery that lets the with-memory schemes get derivative
    information WITHOUT a fresh transcendental call.  Every call to this
    function should be recorded by the caller as a *synthesised* derivative
    (``problem.cost.synthesised_derivatives += 1``), because Section 4.1 of
    the proposal explicitly asks us to distinguish those from real ones.

    Returns the TRUE derivative H^(order)(at), not the Newton coefficient and
    not H^(order)(at) / order!.  The published formulas use H5''(s_k),
    H6'''(v_k) and H7''''(t_k) as genuine derivatives, so getting this
    convention wrong is a silent factor-of-order! error in the parameters.

    ``at`` need not be one of the nodes.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    coefficients = newton_divided_differences(xs, fs, ds)
    if not coefficients:
        raise ValueError("need at least one node to interpolate")

    # Newton form:  H(s) = sum_j c_j * P_j(s),  P_j(s) = prod_{i<j} (s - x_i).
    #
    # Differentiate by carrying the derivatives of the running product rather
    # than expanding powers: with P_{j+1} = P_j * (s - x_j), Leibniz gives
    #     P_{j+1}^(t) = P_j^(t) * (at - x_j) + t * P_j^(t-1).
    # This is exact for polynomials, needs no powers of possibly-tiny node
    # differences, and costs O(m * order).
    #
    # Integer literals keep the accumulator arithmetic-agnostic: int * mpf is
    # an mpf, int * float is a float.
    derivs: list[Any] = [1 if t == 0 else 0 for t in range(order + 1)]
    total = coefficients[0] * derivs[order]

    for j in range(1, len(coefficients)):
        shift = at - xs[j - 1]
        updated: list[Any] = [None] * (order + 1)
        for t in range(order + 1):
            value = derivs[t] * shift
            if t > 0:
                value = value + t * derivs[t - 1]
            updated[t] = value
        derivs = updated
        total = total + coefficients[j] * derivs[order]

    return total


class WithMemoryMixin:
    """Small helpers shared by the with-memory solvers."""

    def _fresh_state(self) -> dict[str, Any]:
        """Initial per-solve state dict wrapping a :class:`MemoryState`."""
        return {"memory": MemoryState()}

    @staticmethod
    def _note_synthesised(problem, count: int = 1) -> None:
        """Tell the cost counter a derivative was interpolated, not evaluated."""
        problem.cost.synthesised_derivatives += count
