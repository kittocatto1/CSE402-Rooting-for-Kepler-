"""The canonical first-order starting guess.

Owner: Mahdi.
"""

from __future__ import annotations

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
        # TODO(Mahdi): implement E0 = M + e * sin(M).
        #   - Evaluate sin(M) with math.sin directly. Do NOT route it through
        #     ``problem`` - the counters in problem.cost measure evaluations at
        #     points E during iteration, and mixing guess cost in there would
        #     corrupt the per-iteration cost table.
        #   - Instead record guess cost in ``self.cost_note`` (see README
        #     "Guess cost" note) if we decide to charge for it.
        raise NotImplementedError("CanonicalGuess: see TODO above")
