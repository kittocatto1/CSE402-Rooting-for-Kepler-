"""The starting guess RadVel itself uses, so our baseline matches production.

Owner: Mahdi.
"""

from __future__ import annotations

from keplerbench.core.base import InitialGuess
from keplerbench.core.registry import register_guess
from keplerbench.core.types import KeplerProblem


@register_guess("radvel")
class RadVelGuess(InitialGuess):
    """Whatever RadVel's ``kepler`` routine uses to seed its Danby iteration.

    This is the guess our "production baseline" (radvel + danby) uses, so it
    must be copied faithfully rather than approximated.
    """

    description = "RadVel production starting guess"

    def __call__(self, problem: KeplerProblem) -> float:
        # TODO(Mahdi): read the starting guess out of RadVel's source and
        # reproduce it EXACTLY here.
        #   1. pip install radvel, then open radvel/kepler.py (function
        #      ``kepler`` / the Cython ``_kepler`` fallback).
        #   2. Copy the expression used to seed the Danby loop, including any
        #      M-range normalisation it does first.
        #   3. Note the exact radvel version in the docstring - if the guess
        #      changes upstream our "production baseline" changes with it.
        raise NotImplementedError("RadVelGuess: see TODO above")
