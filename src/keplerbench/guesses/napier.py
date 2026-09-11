"""Napier (2024) symbolic-regression starting guess.

Reference
---------
K. Napier, "Improved Initial Guesses for Numerical Solutions of Kepler's
Equation", arXiv:2411.15374, 2024.  doi:10.48550/arXiv.2411.15374

This is the guess-layer method under test (Proposal Section 3.2, row 6).

Owner: Mahdi.
"""

from __future__ import annotations

import math

from keplerbench.core.base import InitialGuess
from keplerbench.core.registry import register_guess
from keplerbench.core.types import KeplerProblem

TWO_PI = 2.0 * math.pi


@register_guess("napier")
class NapierGuess(InitialGuess):
    """Napier's elliptical guess, his Equation (2)::

        E0 = e sin M + max{ M, e (sin M + 0.591) }

    That is the expression the paper recommends for the elliptical case over
    the full range e in [0, 1); the terser forms it also reports
    (E = M + e sin M, E = M + 0.71 e) are just the canonical guesses.

    The formula is fitted on e in [0, 1) x M in [0, pi], so M is folded into
    that window first and the answer is mirrored back (``_normalise_M``).
    Nothing divides by (1 - e), so e -> 1 is safe, and at e = 0 it collapses
    to E0 = max{M, 0} = M.
    """

    description = "Napier 2024 symbolic-regression start (arXiv:2411.15374, Eq. 2)"

    def __call__(self, problem: KeplerProblem) -> float:
        e = problem.e
        M_red, sign = self._normalise_M(problem.M)
        s = math.sin(M_red)
        E0 = e * s + max(M_red, e * (s + 0.591))
        # Undo the fold: sign < 0 means M was mirrored about pi.
        folded = E0 if sign > 0 else TWO_PI - E0
        # problem.M - (folded-frame M) is the whole number of turns we dropped.
        turns = problem.M - (M_red if sign > 0 else TWO_PI - M_red)
        return turns + folded

    @staticmethod
    def _normalise_M(M: float) -> tuple[float, int]:
        """Map M into [0, pi] and return (M_reduced, sign) for un-mirroring.

        Kepler's equation has the symmetry E(-M) = -E(M) and
        E(M + 2*pi) = E(M) + 2*pi, so any guess defined on [0, pi] extends to
        the whole line.  ``sign`` is +1 when M already sat in [0, pi] after
        the 2*pi wrap, and -1 when it had to be mirrored about pi.
        """
        M_wrapped = M - TWO_PI * math.floor(M / TWO_PI)
        if M_wrapped > math.pi:
            return TWO_PI - M_wrapped, -1
        return M_wrapped, 1
