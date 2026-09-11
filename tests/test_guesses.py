"""Tests for the starting-guess layer. Owner: Mahdi."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, list_guesses
from keplerbench.core.types import KeplerProblem


def test_simple_guess_is_M():
    g = get_guess("simple")
    assert g(KeplerProblem(e=0.7, M=1.2)) == 1.2


@pytest.mark.parametrize("name", ["simple", "canonical", "radvel", "napier"])
def test_guess_is_exact_at_zero_eccentricity(name):
    """At e = 0 Kepler's equation is E = M, so every sane guess returns M."""
    g = get_guess(name)
    with skip_if_unimplemented():
        assert g(KeplerProblem(e=0.0, M=1.3)) == pytest.approx(1.3, abs=1e-12)


@pytest.mark.parametrize("name", sorted({"simple", "canonical", "radvel", "napier"}))
def test_guess_is_within_one_radian_of_the_root(name):
    """|E - M| <= e < 1, so a guess must land in a window of that size.

    Bound is 1.3 rad, not 1.0: Napier's max-branch deliberately overshoots
    near pericenter at e -> 1, where the measured worst case is
    |E0 - M| = 1.274 at (e, M) = (0.999, 1.046).  That is not a
    transcription error - the overshoot is what buys the iteration count
    the paper reports.  See NapierGuess docstring.
    """
    g = get_guess(name)
    with skip_if_unimplemented():
        for e in (0.1, 0.5, 0.9, 0.99):
            for M in (1e-4, 0.5, 1.5, 3.0):
                E0 = g(KeplerProblem(e=e, M=M))
                assert abs(E0 - M) <= 1.3, (name, e, M, E0)


def test_napier_M_normalisation_respects_symmetry():
    """TODO(Mahdi): once _normalise_M exists, check it against the identity
    E(2*pi - M) = 2*pi - E(M) on a grid."""
    from keplerbench.guesses.napier import NapierGuess

    with skip_if_unimplemented():
        M, _sign = NapierGuess._normalise_M(3.0 * math.pi)
        assert 0.0 <= M <= math.pi
