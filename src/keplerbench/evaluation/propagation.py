"""Downstream metrics: solver error -> orbital parameter error (Section 4.3).

Every function here is pure DataFrame/dict transformation - no fitting, no
solving. That work happens in ``experiments/error_propagation.py`` and
``experiments/monte_carlo.py``; this module only turns their raw output into
the tables the report needs, so every number here traces back to a result
file (the same house rule ``plotting/`` follows).

Owner: Fariha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Fitted parameters this study tracks shifts for, in report order.
TRACKED_PARAMS = ["P", "e", "omega", "K", "gamma", "jitter"]


def _require(df: pd.DataFrame, columns: list[str], who: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{who}: DataFrame is missing {missing}; got {list(df.columns)}")


def parameter_shift(fits: pd.DataFrame, reference_tol: float) -> pd.DataFrame:
    """Shift in each fitted parameter relative to the tightest-tolerance fit.

    For each solver, the row at ``tolerance == reference_tol`` is the
    baseline; every tolerance (including the reference itself, at shift 0)
    is reported as an absolute and relative shift from it. One row per
    (solver, tolerance).
    """
    _require(fits, ["solver", "tolerance"], "parameter_shift")
    params = [p for p in TRACKED_PARAMS if p in fits.columns]
    if not params:
        raise KeyError(
            f"parameter_shift: none of {TRACKED_PARAMS} found in columns {list(fits.columns)}"
        )

    rows = []
    for solver, group in fits.groupby("solver", sort=False):
        # atol=0: np.isclose's default atol=1e-8 would treat every tolerance
        # <= ~1e-8 as "equal" to 1e-14 and silently pick the wrong baseline.
        reference_rows = group.loc[np.isclose(group["tolerance"], reference_tol,
                                              rtol=1e-9, atol=0.0)]
        if reference_rows.empty:
            raise ValueError(
                f"parameter_shift: no row for solver {solver!r} at "
                f"tolerance={reference_tol!r}; available tolerances: "
                f"{sorted(group['tolerance'].unique())}"
            )
        reference = reference_rows.iloc[0]

        for _, row in group.iterrows():
            entry = {"solver": solver, "tolerance": row["tolerance"]}
            for p in params:
                shift = row[p] - reference[p]
                entry[f"{p}_shift"] = shift
                entry[f"{p}_shift_rel"] = shift / reference[p] if reference[p] else float("nan")
            rows.append(entry)
    return pd.DataFrame(rows)


def shift_in_sigma(shifts: pd.DataFrame, posterior_sigma: dict) -> pd.DataFrame:
    """Express each shift as a multiple of the MCMC posterior width.

    This is the number that answers "does the solver choice matter?".
    A shift of 0.01 sigma is irrelevant; a shift of 0.5 sigma is not.
    ``posterior_sigma`` is keyed by the same parameter names as
    :data:`TRACKED_PARAMS` (e.g. the dict returned by
    ``error_propagation.run_reference_mcmc``).
    """
    out = shifts.copy()
    for param in TRACKED_PARAMS:
        shift_col = f"{param}_shift"
        sigma = posterior_sigma.get(param)
        if shift_col in out.columns and sigma:
            out[f"{param}_shift_sigma"] = out[shift_col] / sigma
    return out


def compare_error_budgets(solver_shifts: pd.DataFrame, noise_spread: dict,
                          posterior_sigma: dict) -> pd.DataFrame:
    """The final summary table of the propagation study.

    Columns: parameter | solver-induced shift | noise-driven spread |
             MCMC sigma | ratio(solver / noise)

    ``solver_shifts`` is ``parameter_shift``'s output (or ``shift_in_sigma``'s
    - either has the ``<param>_shift`` columns this needs); the
    solver-induced shift reported per parameter is the largest absolute
    shift seen across every solver and tolerance, i.e. the worst case for
    "does the solver matter". ``noise_spread`` and ``posterior_sigma`` are
    dicts keyed the same way (e.g. ``monte_carlo`` row std devs, and
    ``error_propagation.run_reference_mcmc``'s output).
    """
    _require(solver_shifts, [], "compare_error_budgets")
    available = [p for p in TRACKED_PARAMS if f"{p}_shift" in solver_shifts.columns]

    rows = []
    for param in available:
        solver_shift = float(solver_shifts[f"{param}_shift"].abs().max())
        noise = noise_spread.get(param)
        sigma = posterior_sigma.get(param)
        rows.append({
            "parameter": param,
            "solver_induced_shift": solver_shift,
            "noise_driven_spread": float(noise) if noise is not None else float("nan"),
            "mcmc_sigma": float(sigma) if sigma is not None else float("nan"),
            "ratio_solver_to_noise": (solver_shift / noise) if noise else float("nan"),
        })
    return pd.DataFrame(rows)


def fit_timing(fits: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock per full RadVel fit, one row per solver (Section 4.3).

    Medians over the tolerance sweep, since each (solver, tolerance) fit is
    timed once. ``slowdown_vs_native`` compares against the identical fit
    run with RadVel's compiled solver; ``us_per_solve`` divides by the number
    of Kepler solves, so it is comparable across fits of different length.
    Timings are of this project's pure-Python harness, not of a compiled
    implementation of each method.
    """
    _require(fits, ["solver", "fit_seconds", "native_fit_seconds", "n_solves"], "fit_timing")
    per_fit = fits.assign(
        us_per_solve=1e6 * fits["fit_seconds"] / fits["n_solves"],
        slowdown_vs_native=fits["fit_seconds"] / fits["native_fit_seconds"],
    )
    cols = ["fit_seconds", "native_fit_seconds", "slowdown_vs_native", "us_per_solve", "n_solves"]
    table = per_fit.groupby("solver", sort=False)[cols].median().reset_index()
    return table.rename(columns={c: f"median_{c}" for c in cols})
