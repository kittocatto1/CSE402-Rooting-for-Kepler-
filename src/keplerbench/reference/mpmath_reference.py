"""Ground-truth roots of Kepler's equation at ~50 decimal digits.

Every error metric in the project is measured against these, so this module
must be correct and must NOT reuse any of the five solvers under test.  Use
mpmath at high precision with a bracketing method, which cannot be accused of
favouring any candidate.

Owner: Anisa.
"""

from __future__ import annotations

from functools import lru_cache

# mpmath is a hard dependency - see requirements.txt
import mpmath as mp

#: Working precision in decimal digits. 50 leaves ~35 digits of headroom over
#: the 1e-14 tolerances the solvers are tested at.
REFERENCE_DPS = 50


@lru_cache(maxsize=200_000)
def reference_root(e: float, M: float, dps: int = REFERENCE_DPS) -> float:
    """Return E solving E - e sin E - M = 0, correct to float precision.

    Cached because the grid benchmark asks for the same (e, M) once per
    solver-guess combination, and a high-precision solve is expensive.

    TODO(Anisa):
      1. Set mp.mp.dps = dps inside a context so callers are not affected.
      2. Define g(E) = E - e*sin(E) - M with mpmath types (mp.mpf, mp.sin).
      3. Bracket the root. For M in [0, 2*pi) a valid bracket is
         [M - 1, M + 1] because |E - M| <= e < 1; widen and assert
         g(lo)*g(hi) < 0 so a bad bracket fails loudly instead of silently
         converging somewhere else.
      4. Use mp.findroot(..., solver="anderson"/"bisect") - a BRACKETING
         solver, not a Newton one.
      5. Verify |g(E)| < 1e-40 before returning, else raise.
      6. Return float(E).
    """
    raise NotImplementedError("reference_root: see TODO above")


def reference_roots_grid(e_values, M_values, dps: int = REFERENCE_DPS):
    """Reference roots for a whole (e, M) grid, as a dict keyed by (e, M).

    TODO(Anisa): loop over the grid calling :func:`reference_root`, and
    persist the table to ``results/reference_roots.csv`` so the expensive
    computation is done once for the whole team. Add a ``--rebuild`` flag in
    the CLI to force recomputation.
    """
    raise NotImplementedError("reference_roots_grid: see TODO above")
