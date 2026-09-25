from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from keplerbench.experiments.error_propagation import (
    INITIAL_GUESS,
    INJECTED_TRUTH,
    _orbit_params_from_extra,
    inject_synthetic_dataset,
    noiseless_curve,
)
from keplerbench.io.config import load_config
from keplerbench.io.results_io import results_path
from keplerbench.rv.dataset import load_rv_dataset
from keplerbench.rv.model import OrbitParams
from keplerbench.rv.radvel_bridge import fit_with_solver


def run_monte_carlo(dataset: pd.DataFrame, truth: OrbitParams, jitter: float,
                    n_realisations: int, solver_name: str, tol: float,
                    guess_name: str = "canonical", max_iter: int = 50,
                    seed: int = 0, include_jitter: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = dataset["time"].to_numpy(dtype=float)
    errvel = dataset["errvel"].to_numpy(dtype=float)
    sigma = np.sqrt(errvel**2 + jitter**2) if include_jitter else errvel

    # gamma is NaN after a multi-instrument fit; the injected curve has none
    curve_truth = replace(truth, gamma=0.0) if math.isnan(truth.gamma) else truth
    truth_curve = noiseless_curve(times, curve_truth)

    rows = []
    for i in range(n_realisations):
        realisation = dataset.copy()
        realisation["mnvel"] = truth_curve + rng.normal(0.0, sigma)

        fitted, n_solves, fit_jitter = fit_with_solver(
            realisation, truth, solver_name, tol,
            guess_name=guess_name, max_iter=max_iter,
        )
        rows.append({
            "realisation": i, "P": fitted.P, "tp": fitted.tp, "e": fitted.e,
            "omega": fitted.omega, "K": fitted.K, "gamma": fitted.gamma,
            "jitter": fit_jitter, "n_solves": n_solves,
        })
    return pd.DataFrame(rows)


def run(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    dataset_name = cfg.extra.get("dataset") or "k2-24"
    real_dataset = load_rv_dataset(dataset_name)

    truth = _orbit_params_from_extra(cfg, "injected_truth", INJECTED_TRUTH)
    initial_guess = _orbit_params_from_extra(cfg, "initial_guess", INITIAL_GUESS)
    dataset = inject_synthetic_dataset(real_dataset, truth, seed=cfg.seed)

    solver_name = cfg.solvers[0]
    guess_name = cfg.guesses[0] if cfg.guesses else "canonical"

    best_fit, _, best_jitter = fit_with_solver(
        dataset, initial_guess, solver_name, cfg.tol,
        guess_name=guess_name, max_iter=cfg.max_iter,
    )

    n_realisations = cfg.extra.get("n_realisations", 50)
    include_jitter = cfg.extra.get("include_jitter", True)

    mc_df = run_monte_carlo(
        dataset, best_fit, best_jitter, n_realisations, solver_name, cfg.tol,
        guess_name=guess_name, max_iter=cfg.max_iter, seed=cfg.seed,
        include_jitter=include_jitter,
    )
    mc_df.to_csv(results_path("monte_carlo", "raw.csv"), index=False)

    summary = mc_df[["P", "tp", "e", "omega", "K", "gamma"]].std()
    summary.name = "noise_driven_spread"
    summary.to_csv(results_path("monte_carlo", "summary.csv"))

    return mc_df
