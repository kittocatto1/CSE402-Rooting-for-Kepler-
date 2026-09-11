"""The data objects every solver, experiment and metric passes around.

Owner: Anisa.  Do not change these signatures without telling the team - the
whole pipeline is typed against them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from keplerbench.core.counters import CostCounter


@dataclass
class KeplerProblem:
    """One instance of Kepler's equation:  f(E) = E - e*sin(E) - M = 0.

    Solvers receive this object and evaluate the equation THROUGH it, never by
    calling ``math.sin`` directly.  That is how the cost accounting in
    Section 4.1 of the proposal gets filled in automatically.

    Example
    -------
    >>> p = KeplerProblem(e=0.5, M=0.3)
    >>> f, fp = p.f_fprime(0.4)          # one sincos pair, counted
    """

    #: Eccentricity, 0 <= e < 1 for the elliptical case studied here.
    e: float
    #: Mean anomaly in radians.
    M: float
    #: Cost tally for this solve.
    cost: CostCounter = field(default_factory=CostCounter)

    # ------------------------------------------------------------------
    # Evaluations.  Each one tells the CostCounter what it needed.
    # ------------------------------------------------------------------
    def f(self, E: float) -> float:
        """f(E) = E - e sin E - M.  Needs sin only."""
        self.cost.note_point(E)
        self.cost.sin_only += 1
        return E - self.e * math.sin(E) - self.M

    def fprime(self, E: float) -> float:
        """f'(E) = 1 - e cos E.  Needs cos only."""
        self.cost.note_point(E)
        self.cost.cos_only += 1
        return 1.0 - self.e * math.cos(E)

    def f_fprime(self, E: float) -> tuple[float, float]:
        """(f, f') at one point - one sincos pair, the cheap Kepler case."""
        self.cost.note_point(E)
        self.cost.sincos_pairs += 1
        s = math.sin(E)
        c = math.cos(E)
        return E - self.e * s - self.M, 1.0 - self.e * c

    def derivatives(self, E: float, order: int = 3) -> tuple[float, ...]:
        """(f, f', f'', f''') at one point from a single sincos pair.

        f''  =  e sin E
        f''' =  e cos E
        Danby needs all of these, and they are free once sin/cos are known.
        """
        if order < 1 or order > 3:
            raise ValueError("order must be 1, 2 or 3")
        self.cost.note_point(E)
        self.cost.sincos_pairs += 1
        s = math.sin(E)
        c = math.cos(E)
        out = [E - self.e * s - self.M, 1.0 - self.e * c, self.e * s, self.e * c]
        return tuple(out[: order + 1])

    # ------------------------------------------------------------------
    def reset_cost(self) -> None:
        """Zero the counters. The runner calls this before every solve."""
        self.cost.reset()

    def __repr__(self) -> str:  # keeps log lines short
        return f"KeplerProblem(e={self.e:.6g}, M={self.M:.6g})"


@dataclass
class IterationRecord:
    """One row of the residual history - the raw material for every metric."""

    #: Iteration index, 0 = the starting guess itself.
    iteration: int
    #: Current iterate.
    E: float
    #: |f(E)| at this iterate.
    residual: float
    #: |E - E_reference|, filled in only when a reference root was supplied.
    error: float | None = None
    #: |E_n - E_{n-1}|, always available (used by the ACOC estimator).
    step: float | None = None
    #: Cumulative cost tallies at the end of this iteration.
    cost: dict[str, int] = field(default_factory=dict)


@dataclass
class SolveResult:
    """Everything one solver call produced. Experiments only store these."""

    solver: str
    guess: str
    e: float
    M: float

    #: Final iterate.
    E: float
    #: True if the stopping criterion was met before ``max_iter``.
    converged: bool
    #: Number of iterations actually taken (0 for closed-form solvers).
    iterations: int
    #: Final |f(E)|.
    residual: float
    #: |E - E_reference|, None if no reference root was supplied.
    error: float | None

    #: Per-iteration history, empty when history recording is switched off.
    history: list[IterationRecord] = field(default_factory=list)
    #: Final cumulative cost tallies.
    cost: dict[str, int] = field(default_factory=dict)
    #: Seconds for this single solve (noisy - use the timing harness instead).
    wall_time: float = float("nan")
    #: Set when the solver blew up (overflow, domain error, divergence).
    failure: str | None = None

    def to_row(self) -> dict:
        """Flatten to one dict, ready for a pandas DataFrame / CSV row."""
        row = {
            "solver": self.solver,
            "guess": self.guess,
            "e": self.e,
            "M": self.M,
            "E": self.E,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
            "error": self.error,
            "wall_time": self.wall_time,
            "failure": self.failure,
        }
        row.update({f"cost_{k}": v for k, v in self.cost.items()})
        return row
