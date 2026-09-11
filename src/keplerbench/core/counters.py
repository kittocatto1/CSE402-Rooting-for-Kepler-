"""Kepler-specific cost accounting (Proposal Section 4.1).

Why this file exists
--------------------
Counting "iterations" is not a fair currency for this project.  On Kepler's
equation one ``sincos`` call gives you *both* sin E and cos E for roughly the
price of one transcendental evaluation.  So a method that needs sin and cos at
ONE point is much cheaper than a method that needs sin at one point and cos at
another, even if both "use two evaluations".

Every solver therefore reports:
  1. how many DISTINCT points E it touched per iteration,
  2. how many of those points needed a full (sin, cos) pair vs sin/cos alone,
  3. how many derivatives it faked from divided differences / Hermite
     interpolation instead of paying for a fresh transcendental call.

Owner: Anisa.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostCounter:
    """Tally of the work a solver actually did.

    All counters are cumulative over a whole solve (not per iteration).
    Solvers must NOT touch these fields directly - they go through
    :class:`keplerbench.core.types.KeplerProblem`, which increments them.
    """

    #: Number of distinct E values at which anything was evaluated.
    eval_points: int = 0
    #: Points where both sin E and cos E were needed (one sincos call).
    sincos_pairs: int = 0
    #: Points where only sin E was needed.
    sin_only: int = 0
    #: Points where only cos E was needed.
    cos_only: int = 0
    #: Derivatives obtained from divided differences / Hermite interpolation.
    synthesised_derivatives: int = 0
    #: Cheap arithmetic-only operations a solver wants on the record.
    extra_flops: int = 0

    #: Set of E values already visited, so a repeated point is not double-counted.
    _visited: set[float] = field(default_factory=set, repr=False, compare=False)

    def note_point(self, E: float) -> bool:
        """Record that point ``E`` was touched. Returns True if it is new."""
        is_new = E not in self._visited
        if is_new:
            self._visited.add(E)
            self.eval_points += 1
        return is_new

    def reset(self) -> None:
        """Clear every counter. Called before each solve."""
        self.eval_points = 0
        self.sincos_pairs = 0
        self.sin_only = 0
        self.cos_only = 0
        self.synthesised_derivatives = 0
        self.extra_flops = 0
        self._visited.clear()

    def snapshot(self) -> dict[str, int]:
        """Plain dict of the current tallies (used in per-iteration records)."""
        return {
            "eval_points": self.eval_points,
            "sincos_pairs": self.sincos_pairs,
            "sin_only": self.sin_only,
            "cos_only": self.cos_only,
            "synthesised_derivatives": self.synthesised_derivatives,
            "extra_flops": self.extra_flops,
        }

    def as_dict(self) -> dict[str, int]:
        """Alias of :meth:`snapshot`, for writing results to CSV."""
        return self.snapshot()
