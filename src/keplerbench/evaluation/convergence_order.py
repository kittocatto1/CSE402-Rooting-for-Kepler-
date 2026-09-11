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

Owner: Suchi.
"""

from __future__ import annotations

from keplerbench.core.types import IterationRecord


def coc(errors: list[float]) -> list[float]:
    """Per-step COC estimates from a sequence of absolute errors.

    TODO(Suchi):
      1. Need at least 3 errors to produce one estimate.
      2. Skip steps where any error is 0 or where consecutive errors are
         equal - taking log of 0 or dividing by log(1) both explode. Return
         nan for those positions rather than dropping them, so the caller
         can see where the estimate died.
      3. Return one estimate per usable triple.
    """
    raise NotImplementedError("coc: see TODO above")


def acoc(iterates: list[float]) -> list[float]:
    """Per-step ACOC estimates from the iterate sequence itself."""
    # TODO(Suchi): same structure as coc, but on d_n = |x_n - x_{n-1}|.
    raise NotImplementedError("acoc: see TODO above")


def order_from_history(history: list[IterationRecord],
                       use_reference: bool = True) -> float:
    """Single best order estimate for one solve.

    TODO(Suchi):
      1. Pull errors (if use_reference and record.error is set) or iterates.
      2. Call coc / acoc.
      3. Return the LAST finite estimate before the sequence hits the
         double-precision floor - that is the one in the asymptotic regime.
         Averaging all estimates is wrong here: the early ones are
         pre-asymptotic and the late ones are pure round-off noise.
      4. Return nan if no usable estimate exists, and make sure downstream
         aggregation reports how often that happened.

    IMPORTANT: for a method of order ~10 starting from a good guess, you can
    hit machine precision in 2 iterations, leaving no usable triple at all.
    That is why verification (experiments/verification.py) should run in
    extended precision. On the double-precision Kepler grid, expect the
    measured order for the high-order methods to be unreliable, and SAY SO
    in the report instead of reporting a noisy number.
    """
    raise NotImplementedError("order_from_history: see TODO above")
