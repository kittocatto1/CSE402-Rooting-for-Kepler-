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

Owner: Suchi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryState:
    """Values carried from one iteration to the next.

    Fill in the exact fields once you know which quantities each paper's
    interpolation actually needs - this is a starting shape, not a spec.
    """

    #: Iterates from the previous iteration (x_{n-1} and its sub-steps).
    prev_points: list[float] = field(default_factory=list)
    #: Function values at ``prev_points``, in the same order.
    prev_values: list[float] = field(default_factory=list)
    #: Self-accelerating parameters. NWM9 has one, NWM11 has two.
    params: dict[str, float] = field(default_factory=dict)
    #: True until the first full iteration has run - see note below.
    first_iteration: bool = True

    def has_memory(self) -> bool:
        """Whether enough history exists to compute the parameters."""
        return not self.first_iteration and len(self.prev_points) > 0


def newton_divided_differences(xs: list[float], fs: list[float]) -> list[float]:
    """Divided-difference table for Newton's interpolating polynomial.

    Returns the leading coefficients f[x0], f[x0,x1], f[x0,x1,x2], ...

    Both papers build their self-accelerating parameters out of exactly this,
    so it is factored out here and unit-tested once.

    TODO(Suchi): implement, and test against a polynomial of known degree
    (the divided differences of a degree-k polynomial must vanish above
    order k).
    """
    raise NotImplementedError("newton_divided_differences: see TODO above")


def hermite_derivative_estimate(
    xs: list[float], fs: list[float], order: int, at: float
) -> float:
    """Estimate the ``order``-th derivative of f at ``at`` by Hermite/Newton
    interpolation through the already-computed points.

    This is the machinery that lets the with-memory schemes get derivative
    information WITHOUT a fresh transcendental call.  Every call to this
    function should be recorded by the caller as a *synthesised* derivative
    (``problem.cost.synthesised_derivatives += 1``), because Section 4.1 of
    the proposal explicitly asks us to distinguish those from real ones.

    TODO(Suchi): implement on top of ``newton_divided_differences``.
    """
    raise NotImplementedError("hermite_derivative_estimate: see TODO above")


class WithMemoryMixin:
    """Small helpers shared by NWM9 and NWM11."""

    def _fresh_state(self) -> dict[str, Any]:
        """Initial per-solve state dict wrapping a :class:`MemoryState`."""
        return {"memory": MemoryState()}

    @staticmethod
    def _note_synthesised(problem, count: int = 1) -> None:
        """Tell the cost counter a derivative was interpolated, not evaluated."""
        problem.cost.synthesised_derivatives += count
