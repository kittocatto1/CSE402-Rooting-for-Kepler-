"""Failure and non-convergence rates across the (e, M) grid (Section 4.3).

Owner: Mahdi.
"""

from __future__ import annotations

import pandas as pd


def failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per (solver, guess): fraction of grid points that did not converge.

    TODO(Mahdi): group by solver and guess; report
        n_points, n_converged, n_max_iter_hit, n_exception, failure_rate
    Distinguish the two failure modes - hitting max_iter is "slow", raising
    is "broken", and they mean different things in the report.
    """
    raise NotImplementedError("failure_rates: see TODO above")


def failure_rates_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Failure rates split into the ordinary region and the hard corner.

    TODO(Mahdi): define the regions explicitly (e.g. hard corner is
    e > 0.9 AND M < 0.1) and state the definition in the report. An overall
    average hides exactly the effect the proposal cares about, because the
    corner is a small fraction of any grid.
    """
    raise NotImplementedError("failure_rates_by_region: see TODO above")


def wrong_root_rate(df: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Fraction of solves that converged to the WRONG root.

    A solve with a tiny residual can still be wrong if the iteration jumped
    to a different branch. Compare against the reference root, not the
    residual.

    TODO(Mahdi): count rows where converged is True but error > tol.
    """
    raise NotImplementedError("wrong_root_rate: see TODO above")
