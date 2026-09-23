"""Ground-truth roots of Kepler's equation at ~50 decimal digits.

Every error metric in the project is measured against these, so this module
must be correct and must NOT reuse any of the five solvers under test.  Use
mpmath at high precision with a bracketing method, which cannot be accused of
favouring any candidate.

Owner: Anisa.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Sequence

# mpmath is a hard dependency - see requirements.txt
import mpmath as mp

from keplerbench.io.results_io import REPO_ROOT

#: Working precision in decimal digits. 50 leaves ~35 digits of headroom over
#: the 1e-14 tolerances the solvers are tested at.
REFERENCE_DPS = 50

#: A root is only accepted once |E - e sin E - M| drops below this.  Far
#: tighter than any tolerance the solvers are judged at, so the reference
#: never contributes measurably to a reported error.
RESIDUAL_TOL = "1e-40"

#: Shared on-disk cache of computed roots, so the expensive high-precision
#: pass is paid once for the whole team rather than once per run.  Lives at
#: the top of ``results/`` rather than under an experiment directory because
#: it is not the output of any one experiment.  ``results/`` is gitignored.
CACHE_PATH = REPO_ROOT / "results" / "reference_roots.csv"

_CACHE_COLUMNS = ("e", "M", "E", "dps")


def _bracket(g, M: mp.mpf, max_widen: int = 60) -> tuple[mp.mpf, mp.mpf]:
    """Return (lo, hi) with a sign change of ``g`` between them.

    For 0 <= e < 1 the root always satisfies |E - M| <= e < 1, so
    [M - 1, M + 1] brackets it: g(M - 1) = -1 - e sin(M - 1) <= e - 1 < 0 and
    g(M + 1) = 1 - e sin(M + 1) >= 1 - e > 0.  The widening loop is pure
    defence - if a caller ever passes an e outside the elliptical range, this
    fails loudly instead of letting the solver converge somewhere arbitrary.
    """
    half = mp.mpf(1)
    for _ in range(max_widen):
        lo, hi = M - half, M + half
        if g(lo) * g(hi) <= 0:
            return lo, hi
        half *= 2
    raise RuntimeError(
        f"could not bracket the root around M={M}: no sign change within "
        f"+/-{half}. Is e outside [0, 1)?"
    )


@lru_cache(maxsize=200_000)
def reference_root(e: float, M: float, dps: int = REFERENCE_DPS) -> float:
    """Return E solving E - e sin E - M = 0, correct to float precision.

    Cached because the grid benchmark asks for the same (e, M) once per
    solver-guess combination, and a high-precision solve is expensive.
    """
    if not (0.0 <= e < 1.0):
        raise ValueError(f"e must satisfy 0 <= e < 1 (elliptical case), got {e!r}")
    if not math.isfinite(M):
        raise ValueError(f"M must be finite, got {M!r}")

    # workdps restores the caller's precision on the way out, so importing
    # this module can never silently change someone else's arithmetic.
    with mp.workdps(dps):
        e_hp = mp.mpf(e)
        M_hp = mp.mpf(M)
        tol = mp.mpf(RESIDUAL_TOL)

        def g(E):
            return E - e_hp * mp.sin(E) - M_hp

        lo, hi = _bracket(g, M_hp)

        # Anderson-Bjorck: bracketing, so it cannot wander off the way a
        # Newton step can, but far faster than plain bisection.  verify=False
        # because we check the residual ourselves against RESIDUAL_TOL rather
        # than mpmath's precision-derived default.
        E = None
        for solver in ("anderson", "bisect"):
            try:
                candidate = mp.findroot(
                    g, (lo, hi), solver=solver, tol=tol**2,
                    verify=False, maxsteps=400,
                )
            except (ValueError, ZeroDivisionError):
                continue
            if abs(g(candidate)) <= tol:
                E = candidate
                break

        if E is None:
            raise RuntimeError(
                f"reference_root failed to reach |f| <= {RESIDUAL_TOL} for "
                f"e={e!r}, M={M!r} at dps={dps}"
            )
        return float(E)


def _read_cache(dps: int) -> dict[tuple[float, float], float]:
    """Load previously computed roots for this precision, if any."""
    if not CACHE_PATH.exists():
        return {}
    cached: dict[tuple[float, float], float] = {}
    with CACHE_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                if int(row["dps"]) != dps:
                    # A table computed at a different precision is not
                    # interchangeable with this one; leave it alone.
                    continue
                cached[(float(row["e"]), float(row["M"]))] = float(row["E"])
            except (KeyError, TypeError, ValueError):
                # A truncated or hand-edited cache is a cache miss, never an
                # error - the roots are all recomputable.
                continue
    return cached


def _write_cache(roots: dict[tuple[float, float], float], dps: int) -> None:
    """Merge ``roots`` into the on-disk table and rewrite it.

    Values are written with ``repr`` so a float round-trips exactly: the
    runner looks reference roots up by the (e, M) tuple, and a key that
    survives the CSV only approximately would silently miss.
    """
    merged = _read_cache(dps)
    merged.update(roots)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CACHE_COLUMNS)
        for (e, M), E in sorted(merged.items()):
            writer.writerow([repr(e), repr(M), repr(E), dps])


def reference_roots_grid(
    e_values: Sequence[float],
    M_values: Sequence[float],
    dps: int = REFERENCE_DPS,
    *,
    rebuild: bool = False,
    persist: bool = True,
    progress_every: int = 500,
) -> dict[tuple[float, float], float]:
    """Reference roots for a whole (e, M) grid, as a dict keyed by (e, M).

    ``e_values`` and ``M_values`` are two PARALLEL sequences - one entry each
    per grid point - not two axes to take a cartesian product of.  The
    benchmark grid is a union of a uniform block, a log corner block and a
    random RadVel sample, so it is deliberately not a rectangular lattice.

    Results are cached in :data:`CACHE_PATH` so the expensive computation is
    done once for the whole team.  Pass ``rebuild=True`` to ignore that cache
    and recompute from scratch.
    """
    if len(e_values) != len(M_values):
        raise ValueError(
            "e_values and M_values are parallel sequences, one entry per grid "
            f"point, but got {len(e_values)} and {len(M_values)}"
        )

    points = [(float(e), float(M)) for e, M in zip(e_values, M_values)]
    cached = {} if rebuild else _read_cache(dps)

    roots: dict[tuple[float, float], float] = {}
    computed: dict[tuple[float, float], float] = {}
    for i, point in enumerate(points, start=1):
        if point in roots:
            continue                    # duplicate grid point, already done
        if point in cached:
            roots[point] = cached[point]
            continue
        E = reference_root(point[0], point[1], dps)
        roots[point] = E
        computed[point] = E
        if progress_every and len(computed) % progress_every == 0:
            print(f"  reference roots: {i}/{len(points)} points "
                  f"({len(computed)} computed, rest cached)")

    if persist and computed:
        _write_cache(computed, dps)

    return roots
