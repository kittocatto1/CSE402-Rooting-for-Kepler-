"""Failure and non-convergence rates across the (e, M) grid (Section 4.3).

Owner: Mahdi.
"""

from __future__ import annotations

import pandas as pd

#: The pathological corner, stated once so every table in the report uses the
#: same definition: e > 0.9 AND M < 0.1.  Anisa's aggregate.py imports this.
HARD_CORNER_E = 0.9
HARD_CORNER_M = 0.1


def region(df: pd.DataFrame) -> pd.Series:
    """Label each row "hard_corner" or "ordinary" by the definition above."""
    hard = (df["e"] > HARD_CORNER_E) & (df["M"] < HARD_CORNER_M)
    return pd.Series(["hard_corner" if h else "ordinary" for h in hard],
                     index=df.index, name="region")


def _tally(g: pd.DataFrame) -> pd.Series:
    """Counts for one group. Two failure modes, kept apart on purpose:
    hitting max_iter is "slow", raising is "broken"."""
    n = len(g)
    exception = g["failure"].notna()
    converged = g["converged"].astype(bool)
    n_converged = int(converged.sum())
    return pd.Series({
        "n_points": n,
        "n_converged": n_converged,
        "n_max_iter_hit": int((~converged & ~exception).sum()),
        "n_exception": int(exception.sum()),
        "failure_rate": 1.0 - n_converged / n if n else float("nan"),
    })


def _int_counts(out: pd.DataFrame) -> pd.DataFrame:
    """Counts come back as floats from ``apply``; report them as integers."""
    cols = [c for c in out.columns if c.startswith("n_")]
    return out.astype({c: int for c in cols})


def failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per (solver, guess): fraction of grid points that did not converge."""
    return _int_counts(df.groupby(["solver", "guess"], sort=True)[df.columns.tolist()]
                         .apply(_tally, include_groups=False)
                         .reset_index())


def failure_rates_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Failure rates split into the ordinary region and the hard corner.

    The corner is a small fraction of any grid, so an overall average hides
    exactly the effect the proposal cares about.
    """
    d = df.assign(region=region(df))
    return _int_counts(d.groupby(["solver", "guess", "region"], sort=True)[d.columns.tolist()]
                        .apply(_tally, include_groups=False)
                        .reset_index())


def wrong_root_rate(df: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Fraction of solves that converged to the WRONG root.

    A tiny residual is not proof of correctness - the iteration can land on a
    different branch - so this compares against the reference root, never the
    residual.  Rows with no reference root are excluded from the denominator.
    """
    d = df[df["error"].notna()]

    def tally(g: pd.DataFrame) -> pd.Series:
        converged = g["converged"].astype(bool)
        n = int(converged.sum())
        wrong = int((converged & (g["error"] > tol)).sum())
        return pd.Series({
            "n_converged": n,
            "n_wrong_root": wrong,
            "wrong_root_rate": wrong / n if n else float("nan"),
        })

    return _int_counts(d.groupby(["solver", "guess"], sort=True)[d.columns.tolist()]
                        .apply(tally, include_groups=False)
                        .reset_index())
