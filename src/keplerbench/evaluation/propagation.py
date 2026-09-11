"""Downstream metrics: solver error -> orbital parameter error (Section 4.3).

Owner: Fariha.
"""

from __future__ import annotations

import pandas as pd


def parameter_shift(fits: pd.DataFrame, reference_tol: float) -> pd.DataFrame:
    """Shift in each fitted parameter relative to the tightest-tolerance fit.

    TODO(Fariha): for each solver, take the row at ``reference_tol`` as the
    baseline and report absolute and relative shifts in P, e, K (and any
    other fitted parameter) at every looser tolerance.
    """
    raise NotImplementedError("parameter_shift: see TODO above")


def shift_in_sigma(shifts: pd.DataFrame, posterior_sigma: dict) -> pd.DataFrame:
    """Express each shift as a multiple of the MCMC posterior width.

    This is the number that answers "does the solver choice matter?".
    A shift of 0.01 sigma is irrelevant; a shift of 0.5 sigma is not.

    TODO(Fariha): divide each parameter shift by its posterior sigma and
    return the table. Flag anything above a threshold you choose and justify
    in the report.
    """
    raise NotImplementedError("shift_in_sigma: see TODO above")


def compare_error_budgets(solver_shifts: pd.DataFrame,
                          noise_spread: dict,
                          posterior_sigma: dict) -> pd.DataFrame:
    """The final summary table of the propagation study.

    Columns: parameter | solver-induced shift | noise-driven spread |
             MCMC sigma | ratio(solver / noise)

    TODO(Fariha): assemble from the three inputs. This single table is the
    deliverable for Proposal Section 6, bullet 3.
    """
    raise NotImplementedError("compare_error_budgets: see TODO above")
