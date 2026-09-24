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
TWO sin calls, not one. The pair is timed as exactly that, because it is
exactly what ``KeplerProblem.f_fprime`` does. The only single call that
returns both, ``cmath.exp(1j*x)``, measures about 6 sin calls (see
:func:`fused_sincos_weight`), so no route to a discount exists in CPython.  The proposal's cost argument therefore holds for a
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
import timeit
from typing import Mapping

#: Default weights in units of "one plain sin() call", measured by
#: :func:`measure_weights_with_spread` on 2026-09-25 (median of three full
#: runs, each the fastest of 9 interleaved repeats of 2e6 calls; CPython
#: 3.14.7, Apple Silicon arm64). Range over all repeats: sincos 2.12-2.42,
#: cos 0.92-1.08, synthesised 1.66-1.86. Re-run it on the benchmark machine
#: and put the result and range in the report.
#: The synthesised derivative costs MORE than a sin here: in CPython three
#: bytecode arithmetic ops cost more than one C-level sin call. In C the
#: opposite holds, which is another reason both costings are reported.
DEFAULT_WEIGHTS: dict[str, float] = {
    "sincos_pairs": 2.34,
    "sin_only": 1.0,                          # reference unit by definition
    "cos_only": 1.04,
    "synthesised_derivatives": 1.76,
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


#: Calls per timed statement. Unrolling shares the loop overhead across
#: several calls, so the subtraction below takes a small number from a
#: large one instead of two numbers of the same size.
_SLOTS = 8

#: Each primitive as (statement per slot, baseline statement per slot). The
#: baseline does the same bookkeeping with no call, so their difference is
#: the call alone. ``x{i}`` moves every execution so nothing can be cached,
#: and every result lands in ``a`` so none is dead code.
_PRIMITIVES = {
    "sin_only": ("x{i} += d; a += sin(x{i})", "x{i} += d; a += x{i}"),
    "cos_only": ("x{i} += d; a += cos(x{i})", "x{i} += d; a += x{i}"),
    "sincos_pairs": ("x{i} += d; a += sin(x{i}); a += cos(x{i})",
                     "x{i} += d; a += x{i}; a += x{i}"),
    # A synthesised derivative: one first divided difference (fa - fb) /
    # (xa - xb) from values already in hand - pure arithmetic.
    "synthesised_derivatives": ("x{i} += d; a += (x{i} - fb) / (x{i} - xb)",
                                "x{i} += d; a += x{i}"),
    # Not a counter: the only single call that returns sin AND cos in
    # CPython, exp(ix) = cos x + i sin x. Timed to answer whether a fused
    # sincos is available to us at all.
    "fused_sincos": ("x{i} += d; z = cexp(1j * x{i}); a += z.real; a += z.imag",
                     "x{i} += d; a += x{i}; a += x{i}"),
}

_SETUP = ("from math import sin, cos\n"
          "from cmath import exp as cexp\n"
          "a = 0.0; d = 1e-9; fb = 0.1; xb = 0.2\n"
          + "\n".join(f"x{i} = 0.3 + {i} * 1e-3" for i in range(_SLOTS)))


def _timers() -> dict[str, "tuple[timeit.Timer, timeit.Timer]"]:
    def unroll(slot: str) -> str:
        return "\n".join(slot.format(i=i) for i in range(_SLOTS))
    return {key: (timeit.Timer(unroll(stmt), _SETUP),
                  timeit.Timer(unroll(base), _SETUP))
            for key, (stmt, base) in _PRIMITIVES.items()}


def _measure(keys, n_calls: int, repeats: int
             ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Weights for ``keys`` relative to sin, and their per-repeat spread.

    Every primitive and its baseline is timed once per repeat, interleaved,
    so a burst of machine noise hits all of them rather than one. The weight
    uses the FASTEST time of each (the ``timeit`` convention: noise only ever
    adds time, so the minimum is the best estimate of the true cost); the
    spread is the range of the per-repeat ratios.
    """
    if n_calls <= 0 or repeats <= 0:
        raise ValueError("n_calls and repeats must be positive")
    number = max(1, n_calls // _SLOTS)
    timers = _timers()
    wanted = ["sin_only"] + [k for k in keys if k != "sin_only"]
    runs: dict[str, list[float]] = {k: [] for k in wanted}
    best: dict[str, float] = {}
    for _ in range(repeats):
        for key in wanted:
            stmt, base = timers[key]
            call = stmt.timeit(number) - base.timeit(number)
            runs[key].append(call)
            best[key] = min(best.get(key, math.inf), call)
    if best["sin_only"] <= 0.0 or min(runs["sin_only"]) <= 0.0:
        raise RuntimeError("sin() timed at or below the loop overhead; "
                           "increase n_calls")
    weights = {k: best[k] / best["sin_only"] for k in keys}
    spread = {k: (min(r / u for r, u in zip(runs[k], runs["sin_only"])),
                  max(r / u for r, u in zip(runs[k], runs["sin_only"])))
              for k in keys}
    return weights, spread


def measure_weights_with_spread(n_calls: int = 2_000_000, repeats: int = 9
                                ) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Measured weights and their (min, max) per-repeat range.

    The spread goes in the report, so the reader knows how solid the
    weights are.
    """
    return _measure(tuple(DEFAULT_WEIGHTS), n_calls, repeats)


def fused_sincos_weight(n_calls: int = 2_000_000, repeats: int = 9) -> float:
    """Cost of getting sin AND cos from ONE call, ``cmath.exp(1j*x)``.

    The honest answer to "does CPython give a sincos discount": if this is
    not below ``DEFAULT_WEIGHTS["sincos_pairs"]``, no single-call route to
    the pair is cheaper than two plain calls, and the discount exists only
    in compiled code.
    """
    weights, _ = _measure(("fused_sincos",), n_calls, repeats)
    return weights["fused_sincos"]


def measure_weights(n_calls: int = 2_000_000) -> dict[str, float]:
    """Time the primitive operations to calibrate the weights above.

    Times ``n_calls`` of sin, cos, a (sin, cos) pair and a divided
    difference, each against a baseline with the same bookkeeping and no
    call, normalises by the plain sin time and keeps the fastest of 9
    interleaved repeats. ``sin_only`` is 1.0 by definition. Use
    :func:`measure_weights_with_spread` for the spread.

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

    Digits come from ``metrics.correct_digits_from_error``, the project's
    one rule: -log10(final_error) capped at ``DIGIT_CEILING``, and an exact
    hit (final_error == 0) scores the ceiling. Exact hits are 63-74% of
    converged solves on our grid, so returning nan for them would drop most
    of each solver's best results and bias the ratio upward. Returns nan
    only when the ratio means nothing: a missing, non-finite or negative
    error, or an error of 1 or more (no digit gained).
    """
    from keplerbench.evaluation.metrics import correct_digits_from_error

    digits = correct_digits_from_error(final_error)
    if not math.isfinite(digits) or digits <= 0.0:
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
