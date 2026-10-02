#!/usr/bin/env python3
"""One repaired failure, iteration by iteration.

Picks a hard-corner point where published NWM11 fails and guarded NWM11
converges (from results/safeguard/raw.csv), runs published, memory-off and
guarded NWM11 there, and records for every iteration the error against the
50-digit reference root and the guard ratios |alpha_k f(s_k)| / |f'(s_k)| and
|beta_k f(t_k)| / |f'(s_k)| that the published method used. Writes results/safeguard/trajectory.csv.

This shows the mechanism behind the safeguard on one real case. It does not
by itself say how often the mechanism occurs; results/safeguard/summary.csv
does that.

Owner: Mahdi.
"""

from __future__ import annotations

import pandas as pd

import keplerbench.guesses  # noqa: F401
import keplerbench.solvers  # noqa: F401
import keplerbench.solvers.safeguarded  # noqa: F401
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.runner import solve_one
from keplerbench.io.results_io import results_path
from keplerbench.reference.mpmath_reference import reference_roots_grid

# A hard-corner point from results/safeguard/raw.csv where published NWM11
# fails, and both the memory-off and the guarded version converge.
GUESS, E_ECC = "napier", 0.966402


def pick_point() -> tuple[float, float]:
    raw = pd.read_csv(results_path("safeguard", "raw.csv"),
                      usecols=["solver", "guess", "e", "M", "converged"])
    raw = raw[raw.guess == GUESS]
    table = raw.pivot_table(index=["e", "M"], columns="solver",
                            values="converged", aggfunc="first").astype(bool)
    repaired = table[~table.nwm11 & table.nwm11_guarded & table.nwm11_memoryless]
    repaired = repaired.reset_index()
    row = repaired.iloc[(repaired.e - E_ECC).abs().argmin()]
    return float(row.e), float(row.M)


def run(e: float, M: float) -> pd.DataFrame:
    E_ref = float(next(iter(reference_roots_grid([e], [M]).values())))
    rows = []
    for name in ["nwm11", "nwm11_memoryless", "nwm11_guarded"]:
        solver = get_solver(name)
        ratios: list[float] = []
        beta_ratios: list[float] = []
        if name == "nwm11":
            original = solver._alpha

            def recording_alpha(memory, s, fs, fps, problem, _orig=original):
                alpha, term = _orig(memory, s, fs, fps, problem)
                ratios.append(abs(alpha * fs) / abs(fps) if fps else float("inf"))
                return alpha, term

            solver._alpha = recording_alpha
            original_beta = solver._beta

            def recording_beta(memory, s, v, t, fs, fv, ft, fps, alpha_term, problem,
                               _orig=original_beta):
                beta = _orig(memory, s, v, t, fs, fv, ft, fps, alpha_term, problem)
                beta_ratios.append(abs(beta * ft) / abs(fps) if fps else float("inf"))
                return beta

            solver._beta = recording_beta
        result = solve_one(solver, get_guess(GUESS), e, M, max_iter=50,
                           record_history=True, E_reference=E_ref)
        for k, rec in enumerate(result.history):
            rows.append({"solver": name, "e": e, "M": M, "iteration": k,
                         "error": abs(rec.E - E_ref), "residual": abs(rec.residual),
                         "alpha_ratio": ratios[k - 1] if name == "nwm11" and 0 < k <= len(ratios) else None,
                         "beta_ratio": beta_ratios[k - 1] if name == "nwm11" and 0 < k <= len(beta_ratios) else None,
                         "converged": result.converged})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    e, M = pick_point()
    df = run(e, M)
    path = results_path("safeguard", "trajectory.csv")
    df.to_csv(path, index=False)
    print(f"e = {e}, M = {M}: wrote {path}")
    print(df.groupby("solver").agg(iterations=("iteration", "max"),
                                   converged=("converged", "first")))
