"""The canonical first-order starting guess.

Owner: Mahdi.
"""

from __future__ import annotations

import math

from keplerbench.core.base import InitialGuess
from keplerbench.core.registry import register_guess
from keplerbench.core.types import KeplerProblem


@register_guess("canonical")
class CanonicalGuess(InitialGuess):
    """E0 = M + e sin M  (first term of the series solution).

    Standard textbook start; better than E0 = M at low e, still poor in the
    pathological corner (e -> 1, M -> 0).
    """

    description = "E0 = M + e sin M (first-order series)"

    def __call__(self, problem: KeplerProblem) -> float:
        # math.sin directly, NOT problem.f/... - guess cost must not land in
        # the per-iteration cost counters (runner.solve_one resets them after
        # the guess for exactly this reason).
        return problem.M + problem.e * math.sin(problem.M)
