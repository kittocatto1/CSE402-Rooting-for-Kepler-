from __future__ import annotations

import numpy as np
import pandas as pd

TRACKED_PARAMS = ["P", "e", "omega", "K", "gamma", "jitter"]


def _require(df: pd.DataFrame, columns: list[str], who: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{who}: DataFrame is missing {missing}; got {list(df.columns)}")


def parameter_shift(fits: pd.DataFrame, reference_tol: float) -> pd.DataFrame:
    _require(fits, ["solver", "tolerance"], "parameter_shift")
    params = [p for p in TRACKED_PARAMS if p in fits.columns]
    if not params:
        raise KeyError(
            f"parameter_shift: none of {TRACKED_PARAMS} found in columns {list(fits.columns)}"
        )

    rows = []
    for solver, group in fits.groupby("solver", sort=False):
        # atol=0: the default atol=1e-8 matches every tolerance <= 1e-8
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
    out = shifts.copy()
    for param in TRACKED_PARAMS:
        shift_col = f"{param}_shift"
        sigma = posterior_sigma.get(param)
        if shift_col in out.columns and sigma:
            out[f"{param}_shift_sigma"] = out[shift_col] / sigma
    return out


def compare_error_budgets(solver_shifts: pd.DataFrame, noise_spread: dict,
                          posterior_sigma: dict) -> pd.DataFrame:
    _require(solver_shifts, [], "compare_error_budgets")
    available = [p for p in TRACKED_PARAMS if f"{p}_shift" in solver_shifts.columns]

    def _value(d: dict, key: str) -> float:
        v = d.get(key)
        return float("nan") if v is None else float(v)

    rows = []
    for param in available:
        noise = _value(noise_spread, param)
        sigma = _value(posterior_sigma, param)
        # nothing to compare against (gamma for multi-instrument data, jitter)
        if np.isnan(noise) and np.isnan(sigma):
            continue
        solver_shift = float(solver_shifts[f"{param}_shift"].abs().max())
        rows.append({
            "parameter": param,
            "solver_induced_shift": solver_shift,
            "noise_driven_spread": noise,
            "mcmc_sigma": sigma,
            "ratio_solver_to_noise": solver_shift / noise if noise else float("nan"),
        })
    return pd.DataFrame(rows)


def fit_timing(fits: pd.DataFrame) -> pd.DataFrame:
    _require(fits, ["solver", "fit_seconds", "native_fit_seconds", "n_solves"], "fit_timing")
    per_fit = fits.assign(
        us_per_solve=1e6 * fits["fit_seconds"] / fits["n_solves"],
        slowdown_vs_native=fits["fit_seconds"] / fits["native_fit_seconds"],
    )
    cols = ["fit_seconds", "native_fit_seconds", "slowdown_vs_native", "us_per_solve", "n_solves"]
    table = per_fit.groupby("solver", sort=False)[cols].median().reset_index()
    return table.rename(columns={c: f"median_{c}" for c in cols})
