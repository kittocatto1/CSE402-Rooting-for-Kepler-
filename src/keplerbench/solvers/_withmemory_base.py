"""Shared code for the NWM9 and NWM11 with-memory solvers.

What "with memory" means
------------------------
A normal method forgets everything it computed once an iteration ends.  A
with-memory method keeps those numbers and reuses them to tune itself.  It
fits a polynomial through points it has already evaluated and reads
derivatives off that polynomial, so the tuning costs no new function calls.
Better tuning means a higher convergence order for free.

Why the ``ds`` argument exists
------------------------------
Both papers fit their polynomials through node lists where a node appears
twice, for example

    H5 over [s_k, s_k, t_{k-1}, v_{k-1}, s_{k-1}, s_{k-1}]

A repeated node would make the divided-difference table divide by
(x - x) = 0.  The fix is to use f'(x) there instead, so these functions take
the derivatives in ``ds``.  With ``ds=None`` and no repeats they behave like
ordinary Newton divided differences.

The arithmetic here works with both floats and mpmath numbers.  That matters
because the tables get very ill-conditioned as the nodes crowd together near
the root, so verification runs them at high precision.

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
    """What one iteration hands to the next.

    The three "prev" lists line up with each other and hold the previous
    iteration's nodes: (s, v, t), the value of f at each, and f' where known.
    """

    #: The previous iteration's nodes.
    prev_points: list[float] = field(default_factory=list)
    #: f at each of those nodes, same order.
    prev_values: list[float] = field(default_factory=list)
    #: f' at those nodes, None where we do not have it. Needed because the
    #: polynomials use s twice.
    prev_derivatives: list[float | None] = field(default_factory=list)
    #: The tuning parameters: NWM9 has one, NWM11 has two.
    params: dict[str, float] = field(default_factory=dict)
    #: True until one full iteration has run.
    first_iteration: bool = True

    def has_memory(self) -> bool:
        """True once there is enough history to compute the parameters."""
        return not self.first_iteration and len(self.prev_points) > 0

    def record_iteration(
        self,
        points: Sequence[float],
        values: Sequence[float],
        derivatives: Sequence[float | None] | None = None,
    ) -> None:
        """Save this iteration's nodes, replacing the previous ones.

        Only one iteration is kept - that is all the formulas need. Call this
        at the end of ``step``.
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
    """Equal nodes must sit next to each other.

    [a, a, b] is fine. [a, b, a] is not: it would quietly build a different
    polynomial instead of failing.
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
    """Build the divided-difference table for a Newton polynomial.

    Returns the leading coefficients f[x0], f[x0,x1], f[x0,x1,x2], ...
    Both papers build their tuning parameters from these, so it lives here
    and is tested once.

    Repeated nodes
    --------------
    If ``xs[i] == xs[i+1]`` the usual quotient is 0/0. Use the derivative
    instead: f[x, x] = f'(x). Pass it in ``ds``, a list the same length as
    ``xs`` with None wherever it is not needed. Equal nodes must be adjacent.

    A node may repeat at most twice, because we only have first derivatives.
    Three would need f''. That case raises instead of returning a wrong table.
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
    """Estimate the ``order``-th derivative of f at ``at``.

    Fits a polynomial through points we have already evaluated and reads the
    derivative off it. This is how the solvers get derivatives without paying
    for a new sin or cos call. The caller should count each one with
    ``problem.cost.synthesised_derivatives += 1``, so the cost table can tell
    interpolated derivatives apart from real ones.

    Returns the actual derivative, NOT the Newton coefficient and NOT the
    derivative divided by order!. The papers treat these as real derivatives,
    so the wrong convention here is a silent factor-of-order! error.

    ``at`` does not have to be one of the nodes.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    coefficients = newton_divided_differences(xs, fs, ds)
    if not coefficients:
        raise ValueError("need at least one node to interpolate")

    # The polynomial is H(s) = sum_j c_j * P_j(s) with P_j(s) the product of
    # (s - x_i) for i < j. Rather than expand the powers, carry the running
    # product and its derivatives together. Since P_{j+1} = P_j * (s - x_j),
    # the product rule gives
    #     P_{j+1}^(t) = P_j^(t) * (at - x_j) + t * P_j^(t-1).
    # Exact for polynomials, and it never forms powers of tiny node gaps.
    #
    # The 1 and 0 below are plain ints on purpose: int times mpf is an mpf,
    # int times float is a float, so this works for both.
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
    """Helpers shared by NWM9 and NWM11."""

    def _fresh_state(self) -> dict[str, Any]:
        """Fresh state for one solve, holding an empty MemoryState."""
        return {"memory": MemoryState()}

    @staticmethod
    def _note_synthesised(problem, count: int = 1) -> None:
        """Record that a derivative came from interpolation, not evaluation."""
        problem.cost.synthesised_derivatives += count
