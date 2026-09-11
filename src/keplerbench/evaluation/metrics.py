"""Simple per-solve metrics derived from a raw result row.

Owner: Anisa.
"""

from __future__ import annotations

import math

from keplerbench.core.types import SolveResult


def correct_digits(result: SolveResult) -> float:
    """-log10 of the error against the reference root.

    TODO(Anisa): return nan when result.error is None or non-positive;
    cap at the double-precision floor (~16) so an accidental exact hit does
    not produce an infinite "accuracy".
    """
    raise NotImplementedError("correct_digits: see TODO above")


def iterations_to_tolerance(result: SolveResult, tol: float) -> int | None:
    """First iteration whose residual fell below ``tol``.

    Lets one recorded run answer the "iterations to tolerance" question for
    several tolerances without re-running anything.

    TODO(Anisa): scan result.history; return None if never reached.
    """
    raise NotImplementedError("iterations_to_tolerance: see TODO above")


def efficiency_index(order: float, evaluations_per_iteration: float) -> float:
    """Classical efficiency index E = order ** (1 / evaluations).

    Included because the source papers use it - and because the proposal's
    whole argument is that this generic index does NOT capture Kepler's
    sincos structure. Report it side by side with the weighted cost from
    cost_model.py so the difference is visible.
    """
    if evaluations_per_iteration <= 0:
        return float("nan")
    return math.pow(order, 1.0 / evaluations_per_iteration)
