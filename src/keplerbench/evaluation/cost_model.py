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

What the measurement found
--------------------------
CPython has no sincos intrinsic: ``math.sin`` and ``math.cos`` are two
separate C calls, so in our pure-Python harness a sincos pair costs about
TWO sin calls, not one.  The proposal's cost argument therefore holds for a
C/Cython implementation like RadVel's, where the compiler fuses the pair,
but not for this harness.  Both costings are kept and both belong in the
report:

  * :data:`DEFAULT_WEIGHTS`      - measured here, what our Python run pays
  * :data:`CALL_COUNT_WEIGHTS`   - one transcendental call per sincos pair,
    the costing for a fused-sincos implementation

Owner: Dipit.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Mapping

#: Default weights in units of "one plain sin() call", measured by
#: :func:`measure_weights_with_spread` on 2026-09-24 (median of 7 repeats,
#: 2e6 calls each, CPython 3.14.7, Apple Silicon arm64). Spread (min-max):
#: sincos 2.14-2.58, cos 0.94-1.14, synthesised 1.22-1.39. Re-run it on the
#: benchmark machine and put the result and spread in the report.
#: Note the synthesised derivative costs MORE than a sin here: in CPython a
#: few bytecode arithmetic ops cost as much as one C-level sin call. In C the
#: opposite holds, which is another reason both costings are reported.
DEFAULT_WEIGHTS: dict[str, float] = {
    "sincos_pairs": 2.44,
    "sin_only": 1.0,                          # reference unit by definition
    "cos_only": 1.06,
    "synthesised_derivatives": 1.35,
}

#: Costing for an implementation with a fused sincos (C/Cython, RadVel):
#: each point costs one transcendental call whatever it needs, and a
#: divided-difference derivative costs nothing next to it. Not measured - it
#: is the proposal's counting convention, reported beside the measured one.
CALL_COUNT_WEIGHTS: dict[str, float] = {
    "sincos_pairs": 1.0,
    "sin_only": 1.0,
    "cos_only": 1.0,
    "synthesised_derivatives": 0.0,
}


def _time_loops(n_calls: int) -> dict[str, float]:
    """Seconds per call for each primitive, loop overhead subtracted.

    x changes every iteration so no call can be hoisted out of the loop, and
    every result is folded into an accumulator so none is dead code.
    """
    sin, cos = math.sin, math.cos
    dx = 1e-7

    def empty():
        x, acc = 0.3, 0.0
        for _ in range(n_calls):
            acc += x
            x += dx
        return acc

    def sin_only():
        x, acc = 0.3, 0.0
        for _ in range(n_calls):
            acc += sin(x)
            x += dx
        return acc

    def cos_only():
        x, acc = 0.3, 0.0
        for _ in range(n_calls):
            acc += cos(x)
            x += dx
        return acc

    def sincos_pair():
        x, acc = 0.3, 0.0
        for _ in range(n_calls):
            acc += sin(x) + cos(x)
            x += dx
        return acc

    def divided_difference():
        # What a synthesised derivative costs: one first divided difference
        # (fa - fb) / (a - b) from values already in hand.
        x, fb, acc = 0.3, 0.1, 0.0
        for _ in range(n_calls):
            acc += (x - fb) / (x - 0.2)
            x += dx
        return acc

    loops = {"empty": empty, "sin_only": sin_only, "cos_only": cos_only,
             "sincos_pairs": sincos_pair,
             "synthesised_derivatives": divided_difference}
    seconds = {}
    for key, loop in loops.items():
        t0 = time.perf_counter()
        loop()
        seconds[key] = (time.perf_counter() - t0) / n_calls
    overhead = seconds.pop("empty")
    return {key: max(s - overhead, 0.0) for key, s in seconds.items()}


def measure_weights_with_spread(n_calls: int = 2_000_000, repeats: int = 7
                                ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Median weights and their (min, max) over ``repeats`` runs.

    The spread goes in the report, so the reader knows how solid the
    weights are.
    """
    if n_calls <= 0 or repeats <= 0:
        raise ValueError("n_calls and repeats must be positive")
    runs: dict[str, list[float]] = {}
    for _ in range(repeats):
        per_call = _time_loops(n_calls)
        unit = per_call["sin_only"]
        if unit <= 0.0:
            raise RuntimeError("sin() timed at zero; increase n_calls")
        for key, s in per_call.items():
            runs.setdefault(key, []).append(s / unit)
    weights = {key: statistics.median(v) for key, v in runs.items()}
    spread = {key: (min(v), max(v)) for key, v in runs.items()}
    return weights, spread


def measure_weights(n_calls: int = 2_000_000) -> dict[str, float]:
    """Time the primitive operations to calibrate the weights above.

    Times ``n_calls`` of sin, cos, a (sin, cos) pair and a divided
    difference, subtracts the bare loop overhead, normalises by the plain
    sin time and takes the median of 7 repeats. ``sin_only`` is 1.0 by
    definition. Use :func:`measure_weights_with_spread` for the spread.

    On CPython expect ``sincos_pairs`` close to 2.0: there is no sincos
    intrinsic, so the pair is two separate calls (see the module docstring).
    """
    weights, _ = measure_weights_with_spread(n_calls)
    return weights


def weighted_cost(counts: Mapping[str, int],
                  weights: Mapping[str, float] | None = None) -> float:
    """Total cost of one solve in units of one plain sin() call.

    Keys of ``counts`` not in ``weights`` are ignored (eval_points and
    extra_flops are diagnostics, not costs). A zero count contributes zero
    even when its weight is unmeasured, so a solver never pays for work it
    did not do.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    total = 0.0
    for key, weight in weights.items():
        count = counts.get(key, 0)
        if count:
            total += count * weight
    return total


def cost_per_correct_digit(counts: Mapping[str, int], final_error: float,
                           weights: Mapping[str, float] | None = None) -> float:
    """Weighted cost divided by digits of accuracy gained.

    This is the fairest single-number comparison across methods of different
    order: a method of order 10 that costs 4x per iteration is not obviously
    better than one of order 4, and this ratio settles it.

    digits = -log10(final_error), capped at ``metrics.DIGIT_CEILING`` like
    every other digit count in the project. Returns nan when the ratio means
    nothing: an exact hit (final_error == 0), a missing or non-finite error,
    or an error of 1 or more (no digit gained). Callers must count the exact
    hits separately - they are the best outcome, not a missing one.
    """
    from keplerbench.evaluation.metrics import DIGIT_CEILING

    if final_error is None:
        return float("nan")
    final_error = float(final_error)
    if not math.isfinite(final_error) or final_error <= 0.0:
        return float("nan")
    digits = min(-math.log10(final_error), DIGIT_CEILING)
    if digits <= 0.0:
        return float("nan")
    return weighted_cost(counts, weights) / digits


#: Column order of :func:`per_iteration_cost_table`.
COST_TABLE_COLUMNS = ("method", "eval points", "sincos pairs", "sin only",
                      "cos only", "synthesised derivatives", "theoretical order")

#: Counter name -> table column.
_COUNTER_COLUMNS = {"eval_points": "eval points", "sincos_pairs": "sincos pairs",
                    "sin_only": "sin only", "cos_only": "cos only",
                    "synthesised_derivatives": "synthesised derivatives"}

#: The instance every solver is measured on: eccentric enough that each
#: iterative method takes several iterations, so its steady state is visible
#: past any warm-up step.
_TABLE_PROBLEM = (0.9, 0.5)


def per_iteration_cost_table() -> "list[dict]":
    """The static Section 4.1 table: what each method costs per iteration.

    Runs ONE solve per registered solver with ``record_history=True`` (simple
    guess, e=0.9, M=0.5) and reads the per-iteration deltas out of the
    recorded cost snapshots. Columns are :data:`COST_TABLE_COLUMNS`.

    The row is the steady-state iteration: the with-memory methods' first
    iteration has no recycled values yet and synthesises nothing, so it is
    skipped whenever a later iteration exists. The counts include the shared
    loop's own residual check (one sin per iteration), which every iterative
    solver pays identically. Markley does not iterate, so its row is the
    whole solve.

    This table is MEASURED from the instrumented code, not written by hand -
    that is the point of the counters.
    """
    from keplerbench.core.registry import get_guess, get_solver, list_solvers
    from keplerbench.experiments.runner import solve_one

    e, M = _TABLE_PROBLEM
    guess = get_guess("simple")
    rows = []
    for name in list_solvers():
        solver = get_solver(name)
        result = solve_one(solver, guess, e, M, record_history=True)
        if len(result.history) >= 2:
            steady = result.history[-1].cost
            before = result.history[-2].cost
            per_iter = {k: steady[k] - before.get(k, 0) for k in _COUNTER_COLUMNS}
        else:
            per_iter = {k: result.cost.get(k, 0) for k in _COUNTER_COLUMNS}
        row = {"method": name}
        row.update({col: per_iter[k] for k, col in _COUNTER_COLUMNS.items()})
        row["theoretical order"] = solver.theoretical_order
        rows.append(row)
    return rows
