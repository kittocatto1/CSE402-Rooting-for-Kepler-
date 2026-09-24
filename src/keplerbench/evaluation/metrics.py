"""Simple per-solve metrics derived from a raw result row.

Owner: Anisa.
"""

from __future__ import annotations

import math

from keplerbench.core.types import SolveResult

#: Most digits we are willing to claim from a double-precision run. The root
#: itself is only representable to about 16 significant digits, so anything
#: beyond this is an artefact of the arithmetic, not accuracy the solver
#: earned.
DIGIT_CEILING = 16.0


def correct_digits_from_error(error: float | None) -> float:
    """Digits of agreement with the reference root, from a raw error value.

    Split out from :func:`correct_digits` so that ``aggregate.py`` can apply
    exactly the same rule to an ``error`` column read back from CSV, instead
    of re-deriving it and drifting.

    Three cases, deliberately distinguished:

    * ``None`` or not finite - no reference root was supplied, or the solve
      blew up. The accuracy is unknown, so ``nan``, which drops the row out
      of any mean or median rather than biasing it.
    * exactly ``0.0`` - the iterate landed on the reference root to the last
      bit. That is the best measurable outcome, so it scores the ceiling.
      Returning ``nan`` here would quietly delete a solver's best results
      from the statistics and make it look worse than it is.
    * negative - impossible for an absolute error, so treat it as corrupt
      input and return ``nan`` rather than inventing a number.
    """
    if error is None:
        return float("nan")
    error = float(error)
    if math.isnan(error) or math.isinf(error) or error < 0.0:
        return float("nan")
    if error == 0.0:
        return DIGIT_CEILING
    return min(-math.log10(error), DIGIT_CEILING)


def correct_digits(result: SolveResult) -> float:
    """-log10 of the error against the reference root, capped at the
    double-precision floor."""
    return correct_digits_from_error(result.error)


def iterations_to_tolerance(result: SolveResult, tol: float) -> int | None:
    """First iteration whose residual fell below ``tol``.

    Lets one recorded run answer the "iterations to tolerance" question for
    several tolerances without re-running anything.

    ``None`` means the tolerance was never reached - including the case where
    history was not recorded, which is why a caller must not read ``None`` as
    "took too many steps".

    Uses ``residual <= tol``, matching ``IterativeSolver._converged``, so
    this metric and the solver's own stopping rule can never disagree about
    the same run.
    """
    for record in result.history:
        if record.residual <= tol:
            return record.iteration
    return None


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
