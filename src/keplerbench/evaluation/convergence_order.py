"""Measure how fast a solver converges, instead of trusting the paper.

A paper claims "order 10.7446". That is a checkable statement about how
quickly the error shrinks each step, so we measure it from the numbers the
solver actually produces. A method that claims 10.7446 and delivers 4 is a
result worth reporting.

Two standard estimators:

  COC  needs the true root:
        p ~ ln|e_{n+1}/e_n| / ln|e_n/e_{n-1}|,   e_n = |x_n - x*|

  ACOC uses the gaps between successive iterates instead, so it works
        even when the true root is unknown:
        p ~ ln|d_{n+1}/d_n| / ln|d_n/d_{n-1}|,   d_n = |x_n - x_{n-1}|

Precision
---------
Everything here works with plain floats and with mpmath numbers, and keeps
whatever precision it is given. That matters: an order-10 method uses up
all 16 digits of a double in two steps, which leaves nothing to measure.
See :func:`order_from_history`.

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
# Small helpers so the estimators work on floats and mpmath numbers alike.
# Do NOT swap these for plain math.log: given an mpmath number it quietly
# converts to a double first, throwing away the precision we are paying
# for.
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
    """One order estimate from three errors in a row.

    Returns nan instead of raising when the estimate is undefined, so the
    caller can see exactly where the sequence stopped being useful.
    """
    nan = _nan_like(a)
    for v in (a, b, c):
        if not _isfinite(v) or v <= 0:
            return nan

    # Subtract the logs rather than take log(c/b). These errors run from
    # 1e-2 down past 1e-300, where the ratio underflows to zero but the
    # difference of logs stays a sensible size. The final division is
    # between two ordinary numbers, so nothing is lost either way.
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
    """One order estimate per group of three consecutive errors.

    A run of k errors gives k - 2 estimates. Where an estimate cannot be
    formed - a zero or negative error, a non-finite one, or two equal
    errors in a row, which would divide by log(1) = 0 - the result is nan
    AT THAT POSITION rather than dropped. Keeping the position means the
    index always tells you which three errors produced which estimate.

    Use :func:`last_finite` to pull the final usable estimate out.
    """
    n = len(errors)
    if n < 3:
        return []
    return [_triple_order(errors[i - 1], errors[i], errors[i + 1])
            for i in range(1, n - 1)]


def acoc(iterates: Sequence[float]) -> list[float]:
    """Same estimate, but from the iterates alone - no true root needed.

    Runs :func:`coc` on the gaps between iterates, d_n = |x_n - x_{n-1}|.
    Needs one more input than ``coc`` does, because taking differences
    costs a term: four iterates give three gaps give one estimate.
    """
    n = len(iterates)
    if n < 4:
        return []
    steps = [abs(iterates[i] - iterates[i - 1]) for i in range(1, n)]
    return coc(steps)


def last_finite(estimates: Sequence[float]) -> float | None:
    """The last usable estimate, or None if there is not one.

    Last rather than the average, on purpose. Early estimates come before
    the method has settled into its true rate; late ones, past the
    precision floor, are just rounding noise. Averaging all three phases
    gives a number that describes none of them.
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
    """The measured order for one solve, plus why it can or cannot be used.

    The extra fields exist because ``aggregate.py`` has to report how often
    the measurement failed, and a bare number cannot say that.
    """

    #: The measured order, or nan if there was nothing usable to measure.
    value: float
    #: How many terms were above the precision floor.
    n_usable: int
    #: How many terms the history had in total.
    n_total: int
    #: "error" if measured against the true root, "step" if from the gaps.
    source: str
    #: Plain-English explanation. Always filled in.
    reason: str

    def ok(self) -> bool:
        return not _isnan(self.value)


def _working_floor(scale, like=None) -> float:
    """The smallest error that still means anything.

    Two things set this limit, whichever bites first:

    * the arithmetic itself cannot hold a difference smaller than its own
      epsilon, and
    * the reference root is only known to so many digits, so an "error"
      below |root| * 10**-dps is really measuring the error in the
      reference, not in the iterate.

    The second one caused a real bug. A fast method can overshoot the
    reference root in a single step, and the error then keeps GETTING
    SMALLER while meaning nothing - so the stagnation check below cannot
    spot it. Against a 2000-digit root, an order-8 method came back as 3.38.

    Two digits of margin are left above the limit.

    ``like`` is any value from the sequence, used only to work out which
    kind of arithmetic we are in when ``scale`` cannot say. That matters:
    with a root at exactly zero, or an empty history, this used to fall back
    to a plain 1 and return a double-precision floor of 2.2e-14 even inside
    a 2000-digit run - cutting the sequence off about 1980 orders of
    magnitude too early and calling the whole run unmeasurable.
    """
    working_in_mp = _is_mp(scale) or _is_mp(like)
    one = mp.mpf(1) if working_in_mp else 1.0

    if scale is None or not _isfinite(scale) or scale == 0:
        magnitude = one
    else:
        magnitude = abs(scale)
        if working_in_mp and not _is_mp(magnitude):
            magnitude = mp.mpf(magnitude)

    if working_in_mp:
        return magnitude * mp.mpf(10) ** (-mp.mp.dps + 2)
    # float: 2.22e-16 epsilon, same two digits of guard.
    return float(magnitude) * 2.220446049250313e-14


def _reference_scale(iterates: Sequence[float], magnitudes: Sequence[float]):
    """Best guess at how big the root is, safe against a run that blows up.

    The floor is relative to the root, so we need its size. The obvious
    choice - the last iterate - is wrong for a run that converges and then
    diverges: a single iterate of 1e64 puts the floor at 2.2e50, above every
    real error, and a clean order-2 run gets reported as unmeasurable.
    Solvers really do this in the pathological corner.

    Use the iterate that came CLOSEST to the root instead. That is the best
    evidence the history holds about the root's size. For a well-behaved run
    it is the last iterate anyway, so nothing changes.
    """
    best = None
    best_magnitude = None
    for iterate, magnitude in zip(iterates, magnitudes):
        if not _isfinite(iterate) or not _isfinite(magnitude):
            continue
        if best_magnitude is None or magnitude < best_magnitude:
            best, best_magnitude = iterate, magnitude
    return best


def _usable_prefix(magnitudes: Sequence[float], floor=None) -> list[float]:
    """The part of the sequence that is actually worth measuring.

    Two things need trimming, from opposite ends.

    At the END: once the method hits the precision floor the error stops
    telling us anything - it stalls, bounces, hits exactly zero, or keeps
    shrinking while only measuring the reference root's own error. ``floor``
    cuts that off. The stagnation test alone cannot catch the last case,
    because the numbers do keep falling.

    At the START: the method has not settled into its true rate yet. In
    Kepler's hard corner f'(E) = 1 - e cos E is almost zero, so the first
    step often overshoots and the error goes UP before it comes down. An
    earlier version stopped at the first increase, which threw away the
    whole useful tail whenever that happened: Newton measured a clean 2.0000
    at e = 0.3 and "unmeasurable" at e = 0.9, purely as an artefact.

    So: cut everything from the floor onwards, then keep the longest run of
    steadily-shrinking errors that ENDS at the last surviving term. That run
    is the only part an order estimate describes.
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
    """Measure the order for one solve, and say how much to trust it.

    Uses the true errors against the reference root when they are there,
    and falls back to the gaps between iterates when they are not.

    ``floor`` is the smallest error still worth believing. Pass it if you
    know how accurate the reference root is; otherwise it is worked out
    from the precision in use - see :func:`_working_floor`.
    """
    n_total = len(history)

    have_errors = bool(history) and all(
        record.error is not None for record in history
    )
    if use_reference and have_errors:
        source = "error"
        magnitudes = [abs(record.error) for record in history]
        # Each error belongs to the iterate on the same row.
        aligned_iterates = [record.E for record in history]
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
        # Gap n is |x_n - x_{n-1}|, so it belongs to x_n.
        aligned_iterates = iterates[1:]

    if floor is None:
        # Scale the floor to the root. The errors are absolute, so what
        # counts as "too small to believe" depends on how big the root is.
        scale = _reference_scale(aligned_iterates, magnitudes)
        floor = _working_floor(scale, like=magnitudes[0] if magnitudes else None)

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
    """One order estimate for one solve, or nan if it cannot be measured.

    A thin wrapper around :func:`order_from_history_detail`. Use that one
    when you need to know WHY an estimate is missing.

    IMPORTANT: an order-10 method from a decent starting guess reaches
    machine precision in two steps, which leaves too few points to measure
    anything. That is why verification runs at high precision. On the
    ordinary double-precision grid the measured order for the fast methods
    is unreliable, and the report should SAY SO rather than quote a noisy
    number.
    """
    return order_from_history_detail(
        history, use_reference=use_reference, floor=floor
    ).value
