"""The trivial starting guess, E0 = M.

WORKED EXAMPLE - this one is fully implemented on purpose, so the rest of the
team has a template for the interface.  Copy this shape for the others.

Owner: Mahdi.
"""

from __future__ import annotations

from keplerbench.core.base import InitialGuess
from keplerbench.core.registry import register_guess
from keplerbench.core.types import KeplerProblem


@register_guess("simple")
class SimpleGuess(InitialGuess):
    """E0 = M.

    The cheapest possible start: it costs nothing and is exact at e = 0.
    It is the control condition for the guess-layer factor.
    """

    description = "E0 = M (zeroth-order, free)"

    def __call__(self, problem: KeplerProblem) -> float:
        return problem.M
