"""Kepler-specific cost accounting (Proposal Section 4.1).

This module turns the raw counters in ``CostCounter`` into the single
comparable number that the report's headline comparison uses.

The weighting question, stated plainly
--------------------------------------
On this equation one ``sincos`` call returns sin E and cos E together for
roughly the price of one transcendental evaluation.  So the natural unit is
"transcendental calls", and:

    cost = w_sincos * sincos_pairs
         + w_sin    * sin_only
         + w_cos    * cos_only
         + w_synth  * synthesised_derivatives

with w_sincos only slightly above w_sin, NOT twice it.  The exact weights
must be MEASURED on the machine we benchmark on, not assumed - that
measurement is part of this module's job.

Owner: Dipit.
"""

from __future__ import annotations

from typing import Mapping

#: Default weights in units of "one plain sin() call". Replace with the
#: values measured by :func:`measure_weights` and record them in the report.
DEFAULT_WEIGHTS: dict[str, float] = {
    "sincos_pairs": float("nan"),            # TODO(Dipit): measure
    "sin_only": 1.0,                          # reference unit by definition
    "cos_only": float("nan"),                 # TODO(Dipit): measure
    "synthesised_derivatives": float("nan"),  # TODO(Dipit): measure
}


def measure_weights(n_calls: int = 2_000_000) -> dict[str, float]:
    """Time the primitive operations to calibrate the weights above.

    TODO(Dipit):
      1. Time n_calls of: math.sin(x); math.cos(x); (math.sin(x), math.cos(x))
         as a pair; and a few divided-difference arithmetic ops.
      2. Guard against the compiler/interpreter hoisting the call out of the
         loop - vary x each iteration.
      3. Normalise everything by the plain-sin time.
      4. Repeat and take the median; report the spread in the report so the
         reader knows how solid the weights are.
      5. Note honestly whether CPython actually gives a discount for
         computing sin and cos together. If it does NOT (CPython has no
         sincos intrinsic), say so - then the proposal's cost argument holds
         for a C/Cython implementation like RadVel's but not for our pure
         Python harness, and BOTH costings should appear in the report.
    """
    raise NotImplementedError("measure_weights: see TODO above")


def weighted_cost(counts: Mapping[str, int],
                  weights: Mapping[str, float] | None = None) -> float:
    """Total cost of one solve in units of one plain sin() call.

    TODO(Dipit): dot-product of counts and weights, ignoring keys not in
    ``weights`` (eval_points and extra_flops are diagnostics, not costs).
    """
    raise NotImplementedError("weighted_cost: see TODO above")


def cost_per_correct_digit(counts: Mapping[str, int], final_error: float,
                           weights: Mapping[str, float] | None = None) -> float:
    """Weighted cost divided by digits of accuracy gained.

    This is the fairest single-number comparison across methods of different
    order: a method of order 10 that costs 4x per iteration is not obviously
    better than one of order 4, and this ratio settles it.

    TODO(Dipit): digits = -log10(final_error); return weighted_cost / digits.
    Handle final_error <= 0 (exact hit) by returning nan and counting those
    separately.
    """
    raise NotImplementedError("cost_per_correct_digit: see TODO above")


def per_iteration_cost_table() -> "list[dict]":
    """The static Section 4.1 table: what each method costs per iteration.

    TODO(Dipit): for each solver, run ONE solve with record_history=True and
    read the per-iteration deltas out of the recorded cost snapshots. Build
    a table with columns:

        method | eval points | sincos pairs | sin only | cos only |
        synthesised derivatives | theoretical order

    This table is MEASURED from the instrumented code, not written by hand -
    that is the point of the counters. It goes straight into the report as
    the cost-accounting table.
    """
    raise NotImplementedError("per_iteration_cost_table: see TODO above")
