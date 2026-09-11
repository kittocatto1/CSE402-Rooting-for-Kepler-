"""The starting guess RadVel itself uses, so our baseline matches production.

Owner: Mahdi.
"""

from __future__ import annotations

import math

from keplerbench.core.base import InitialGuess
from keplerbench.core.registry import register_guess
from keplerbench.core.types import KeplerProblem


@register_guess("radvel")
class RadVelGuess(InitialGuess):
    """RadVel's seed for its Danby iteration:  E0 = M + sign(sin M) * 0.85 * e.

    Copied verbatim from radvel 1.6.4, ``radvel/kepler.py``::

        conv = 1.0e-12  # convergence criterion
        k = 0.85
        Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E

    RadVel does no M normalisation before this line, so neither do we.
    If upstream changes the seed, our "production baseline" changes with it -
    re-check this file against the radvel version pinned in pyproject.toml.
    """

    description = "RadVel 1.6.4 production start: E0 = M + sign(sin M)*0.85*e"

    #: RadVel's tuning constant k.
    K = 0.85

    def __call__(self, problem: KeplerProblem) -> float:
        s = math.sin(problem.M)
        sign = math.copysign(1.0, s) if s != 0.0 else 0.0  # np.sign(0) == 0
        return problem.M + sign * self.K * problem.e
