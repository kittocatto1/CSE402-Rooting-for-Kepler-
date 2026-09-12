"""Empirical convergence order, measured - not taken from the paper.

The proposal is explicit about this: we measure the order from the residual
sequence we actually observe.  A claimed order of 10.7446 that shows up as
4 in practice is itself a result worth reporting.

Two standard estimators:

  COC  (computational order of convergence) - needs the true root:
        p ~ ln|e_{n+1}/e_n| / ln|e_n/e_{n-1}|,   e_n = |x_n - x*|

  ACOC (approximated COC) - uses successive differences instead of the true
        root, so it works when x* is unknown:
        p ~ ln|d_{n+1}/d_n| / ln|d_n/d_{n-1}|,   d_n = |x_n - x_{n-1}|

Precision
---------
Every function here is arithmetic-agnostic: pass Python floats and you get
floats back, pass ``mpmath.mpf`` and the whole computation stays at the
working precision of the mpmath context.  That matters because a method of
order ~10 exhausts double precision in two iterations, leaving no usable
triple at all - see :func:`order_from_history`.

Owner: Suchi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import mpmath as mp

from keplerbench.core.types import IterationRecord

__all__ = [
    "coc",
    "acoc",
    "last_finite",
    "order_from_history",
    "order_from_history_detail",
    "OrderEstimate",
]


# ----------------------------------------------------------------------
# Arithmetic helpers.  These exist so the estimators work unchanged on
# float and on mpmath.mpf.  Do NOT replace them with math.log: math.log on
# an mpf silently narrows to double via __float__, which destroys exactly
# the precision the verification step is paying for.
# ----------------------------------------------------------------------
def _is_mp(x: object) -> bool:
    return isinstance(x, (mp.mpf, mp.mpc))


def _log(x):
    return mp.log(x) if _is_mp(x) else math.log(x)


def _nan_like(x):
    return mp.nan if _is_mp(x) else float("nan")


def _isfinite(x) -> bool:
    if _is_mp(x):
        return bool(mp.isfinite(x))
    try:
        return math.isfinite(x)
    except (TypeError, OverflowError):
        return False


def _isnan(x) -> bool:
    if _is_mp(x):
        return bool(mp.isnan(x))
    try:
        return math.isnan(x)
    except (TypeError, OverflowError):
        return False


def _triple_order(a, b, c):
    """One order estimate from three consecutive error magnitudes.

    Returns nan rather than raising whenever the estimate is undefined, so
    the caller can see *where* the sequence stopped being informative.
    """
    nan = _nan_like(a)
    for v in (a, b, c):
        if not _isfinite(v) or v <= 0:
            return nan

    # log(c) - log(b) rather than log(c / b): these sequences span 1e-2 down
    # to 1e-300 and beyond, where the ratio underflows to 0 (or overflows)
    # while the difference of logs stays perfectly well scaled.  The outer
    # quotient is a ratio of two O(1)-O(1000) numbers, so there is no
    # cancellation problem to trade against.
    numerator = _log(c) - _log(b)
    denominator = _log(b) - _log(a)

    if denominator == 0 or not _isfinite(denominator) or not _isfinite(numerator):
        return nan

    p = numerator / denominator
    return p if _isfinite(p) else nan


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def coc(errors: Sequence[float]) -> list[float]:
    """Per-step COC estimates from a sequence of absolute errors.

    One estimate per usable triple ``(e_{n-1}, e_n, e_{n+1})``, so a run of
    ``k`` errors yields ``k - 2`` estimates.  Positions where the estimate is
    undefined - a zero or negative error, a non-finite value, or two equal
    consecutive errors (which makes the denominator ``log(1) = 0``) - come
    back as ``nan`` **in place** rather than being dropped, so the index of
    an estimate always identifies which triple produced it.

    Use :func:`last_finite` to get the asymptotic estimate out of the result.
    """
    n = len(errors)
    if n < 3:
        return []
    return [_triple_order(errors[i - 1], errors[i], errors[i + 1])
            for i in range(1, n - 1)]


def acoc(iterates: Sequence[float]) -> list[float]:
    """Per-step ACOC estimates from the iterate sequence itself.

    Identical estimator to :func:`coc`, applied to the step magnitudes
    ``d_n = |x_n - x_{n-1}|``.  Needs one more input than ``coc`` (four
    iterates give three steps give one estimate) because the differencing
    costs a term.
    """
    n = len(iterates)
    if n < 4:
        return []
    steps = [abs(iterates[i] - iterates[i - 1]) for i in range(1, n)]
    return coc(steps)


def last_finite(estimates: Sequence[float]) -> float | None:
    """The last usable estimate in a run, or None if there is none.

    This is deliberately *last* rather than an average.  Early estimates are
    pre-asymptotic (the method has not settled into its order yet) and late
    ones, once the sequence reaches the precision floor, are pure round-off
    noise.  Averaging the three regimes together produces a number that
    describes none of them.
    """
    for value in reversed(estimates):
        if _isfinite(value) and not _isnan(value):
            return value
    return None


# ----------------------------------------------------------------------
# Single best estimate for one solve
# ----------------------------------------------------------------------
@dataclass
class OrderEstimate:
    """The order measured for one solve, plus why it is (or is not) usable.

    ``aggregate.py`` needs the diagnostic fields to report how often the
    measurement failed; a bare float cannot carry that.
    """

    #: Measured order, or nan when no usable triple existed.
    value: float
    #: How many terms of the sequence were in the pre-floor regime.
    n_usable: int
    #: How many terms the history contained in total.
    n_total: int
    #: "error" (measured against the reference root) or "step" (ACOC).
    source: str
    #: Human-readable explanation, always set.
    reason: str

    def ok(self) -> bool:
        return not _isnan(self.value)


def _working_floor(scale) -> float:
    """Smallest error magnitude that still means something.

    Two different limits, whichever is active:

    * the arithmetic cannot represent a relative difference below its own
      epsilon, and
    * the REFERENCE ROOT is itself only known to the working precision, so
      ``|x_n - xi|`` below ``|xi| * 10**-dps`` is measuring the error in
      ``xi``, not in ``x_n``.

    The second one bites in exactly the regime this project cares about.  A
    high-order method can overshoot the reference root's own accuracy in a
    single step, and the resulting "error" keeps *decreasing* - so the
    stagnation test below cannot see it - while being pure noise.  Measured
    against a 2000-digit root, an order-8 method reported 3.38 instead of
    8.00 for precisely this reason.

    Two digits of guard are left above the limit.
    """
    magnitude = abs(scale) if scale else 1
    if _is_mp(magnitude):
        return magnitude * mp.mpf(10) ** (-mp.mp.dps + 2)
    # float: 2.22e-16 epsilon, same two digits of guard.
    return float(magnitude) * 2.220446049250313e-14


def _usable_prefix(magnitudes: Sequence[float], floor=None) -> list[float]:
    """The asymptotic window of an error sequence.

    Two different things have to be trimmed, from opposite ends.

    At the TAIL, once a method reaches the precision floor the error stops
    being informative - it stalls, bounces, drops to exactly zero, or (against
    a finite-precision reference root) keeps shrinking while measuring nothing
    but the reference's own error.  ``floor`` cuts that off; the stagnation
    test alone cannot see the last case, because the numbers keep falling.

    At the HEAD, a method is not yet converging at its asymptotic rate.  In
    Kepler's pathological corner f'(E) = 1 - e cos E is nearly zero, so the
    first step routinely overshoots and the error goes UP before it comes
    down.  An earlier version of this function stopped at the first
    non-decrease, which threw away the entire asymptotic tail whenever that
    happened - Newton measured a clean 2.0000 at e = 0.3 and "unmeasurable"
    at e = 0.9, purely as an artefact.

    So: drop everything from the floor onwards, then keep the longest
    strictly-decreasing run that ENDS at the last surviving term.  That run is
    the asymptotic regime, which is the only part an order estimate describes.
    """
    kept: list[float] = []
    for value in magnitudes:
        if not _isfinite(value) or value <= 0:
            break
        if floor is not None and value < floor:
            break
        kept.append(value)

    start = len(kept) - 1
    while start > 0 and kept[start] < kept[start - 1]:
        start -= 1
    return kept[start:]


def order_from_history_detail(
    history: Sequence[IterationRecord],
    use_reference: bool = True,
    floor=None,
) -> OrderEstimate:
    """Measure the order for one solve and report how reliable it is.

    Prefers true errors against the reference root when they are available
    (COC), and falls back to step magnitudes (ACOC) otherwise.

    ``floor`` is the smallest magnitude still worth believing; pass it when
    you know the reference root's accuracy, otherwise it is inferred from the
    working precision - see :func:`_working_floor`.
    """
    n_total = len(history)

    have_errors = bool(history) and all(
        record.error is not None for record in history
    )
    if use_reference and have_errors:
        source = "error"
        magnitudes = [abs(record.error) for record in history]
    else:
        source = "step"
        iterates = [record.E for record in history]
        if len(iterates) < 2:
            return OrderEstimate(
                value=float("nan"), n_usable=0, n_total=n_total, source=source,
                reason="history too short to form a single step",
            )
        magnitudes = [abs(iterates[i] - iterates[i - 1])
                      for i in range(1, len(iterates))]

    if floor is None:
        # Scale the floor by the root itself: the errors are absolute, so the
        # meaningful limit is relative to where the root sits.
        scale = history[-1].E if history else None
        floor = _working_floor(scale)

    usable = _usable_prefix(magnitudes, floor=floor)
    estimates = coc(usable)
    best = last_finite(estimates)

    if best is None:
        return OrderEstimate(
            value=float("nan"), n_usable=len(usable), n_total=n_total,
            source=source,
            reason=(
                f"only {len(usable)} term(s) before the precision floor; "
                "at least 3 are needed for one estimate. For a high-order "
                "method this is expected in double precision - re-run the "
                "verification step at higher working precision."
            ),
        )

    return OrderEstimate(
        value=best, n_usable=len(usable), n_total=n_total, source=source,
        reason=f"{len(estimates)} estimate(s) from {len(usable)} pre-floor terms",
    )


def order_from_history(
    history: Sequence[IterationRecord],
    use_reference: bool = True,
    floor=None,
) -> float:
    """Single best order estimate for one solve, or nan if unmeasurable.

    Thin wrapper over :func:`order_from_history_detail`; use that one when
    you need to know *why* an estimate is missing.

    IMPORTANT: for a method of order ~10 starting from a good guess, you can
    hit machine precision in 2 iterations, leaving no usable triple at all.
    That is why verification (experiments/verification.py) runs in extended
    precision.  On the double-precision Kepler grid, expect the measured
    order for the high-order methods to be unreliable, and SAY SO in the
    report instead of reporting a noisy number.
    """
    return order_from_history_detail(
        history, use_reference=use_reference, floor=floor
    ).value
