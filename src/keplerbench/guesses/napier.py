"""Napier (2024) symbolic-regression starting guess.

Reference
---------
K. Napier, "Analytic Starting Guesses for Kepler's Equation via Symbolic
Regression", arXiv:2411.15374, 2024.  doi:10.48550/arXiv.2411.15374

This is the guess-layer method under test (Proposal Section 3.2, row 6).

Owner: Mahdi.
"""

from __future__ import annotations

from keplerbench.core.base import InitialGuess
from keplerbench.core.registry import register_guess
from keplerbench.core.types import KeplerProblem


@register_guess("napier")
class NapierGuess(InitialGuess):
    """Closed-form starting formula found by symbolic regression.

    Drop-in replacement for simple / canonical / radvel starts, used in front
    of EVERY solver so the guess effect is isolated.
    """

    description = "Napier 2024 symbolic-regression start (arXiv:2411.15374)"

    def __call__(self, problem: KeplerProblem) -> float:
        # TODO(Mahdi): transcribe the starting formula from the paper.
        #   1. Get the paper (arXiv:2411.15374) and find the recommended
        #      expression - the paper gives more than one candidate, pick the
        #      one it recommends for the full e-range and say WHICH in this
        #      docstring.
        #   2. Watch the domain: the published formulas usually assume
        #      M in [0, pi]. Normalise M into that range here and mirror the
        #      answer back for M in (pi, 2pi). See ``_normalise_M`` below.
        #   3. Guard the e -> 1 limit; check no division by (1 - e) blows up.
        #   4. Sanity check: at e = 0 the guess must reduce to E0 = M.
        raise NotImplementedError("NapierGuess: see TODO above")

    @staticmethod
    def _normalise_M(M: float) -> tuple[float, int]:
        """Map M into [0, pi] and return (M_reduced, sign) for un-mirroring.

        Kepler's equation has the symmetry E(-M) = -E(M) and
        E(M + 2*pi) = E(M) + 2*pi, so any guess defined on [0, pi] extends to
        the whole line.
        """
        # TODO(Mahdi): implement the 2*pi wrap and the [pi, 2pi] mirror, and
        # unit-test it in tests/test_guesses.py against a brute-force check.
        raise NotImplementedError("NapierGuess._normalise_M: see TODO above")
