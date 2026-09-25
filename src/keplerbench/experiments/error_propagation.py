from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from keplerbench.evaluation.propagation import fit_timing, parameter_shift, shift_in_sigma
from keplerbench.io.config import ExperimentConfig, load_config
from keplerbench.io.results_io import results_path
from keplerbench.rv.anomaly import mean_anomaly, radial_velocity, true_anomaly
from keplerbench.rv.dataset import load_rv_dataset
from keplerbench.rv.model import OrbitParams
from keplerbench.rv.radvel_bridge import fit_with_solver

# Synthetic, moderately eccentric orbit injected on real observation times
# and uncertainties (a single-planet fit of the real multi-planet signal is
# near-circular, where the Kepler solve barely matters). Not a claim about
# any real planet. run() reads the same values from the YAML config.
INJECTED_TRUTH = OrbitParams(P=20.885258, tp=2400.0, e=0.35, omega=1.0, K=5.0, gamma=0.0)

INITIAL_GUESS = OrbitParams(P=INJECTED_TRUTH.P, tp=INJECTED_TRUTH.tp,
                            e=0.15, omega=0.3, K=3.0, gamma=0.0)


def _orbit_params_from_extra(cfg: ExperimentConfig, key: str,
                             default: OrbitParams) -> OrbitParams:
    spec = cfg.extra.get(key)
    return default if spec is None else OrbitParams(**spec)


def _reference_E(e: float, M: float) -> float:
    # bracketing solve, independent of the five solvers under test
    f = lambda E: E - e * np.sin(E) - M
    return brentq(f, M - 1.2, M + 1.2, xtol=1e-15, rtol=8.881784197001252e-16)


def noiseless_curve(times, truth: OrbitParams) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    velocities = np.empty(times.size)
    for i, t in enumerate(times):
        M = mean_anomaly(t, truth.P, truth.tp)
        E = _reference_E(truth.e, M)
        nu = true_anomaly(E, truth.e)
        velocities[i] = radial_velocity(nu, truth.K, truth.e, truth.omega, truth.gamma)
    return velocities


def inject_synthetic_dataset(real_dataset: pd.DataFrame, truth: OrbitParams,
                             seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    errvel = real_dataset["errvel"].to_numpy(dtype=float)
    velocities = noiseless_curve(real_dataset["time"], truth) + rng.normal(0.0, errvel)

    out = real_dataset.copy()
    out["mnvel"] = velocities
    return out


def run_tolerance_sweep(dataset: pd.DataFrame, initial_guess: OrbitParams,
                        solvers: list[str], tolerances: list[float],
                        guess_name: str = "canonical", max_iter: int = 50,
                        vary_period: bool = True,
                        fixed_tc: float | None = None) -> pd.DataFrame:
    rows = []
    for solver_name in solvers:
        for tol in tolerances:
            timing: dict[str, float] = {}
            try:
                fitted, n_solves, jitter = fit_with_solver(
                    dataset, initial_guess, solver_name, tol,
                    guess_name=guess_name, max_iter=max_iter,
                    vary_period=vary_period, fixed_tc=fixed_tc,
                    timing_out=timing,
                )
            except NotImplementedError as exc:
                warnings.warn(f"skipping {solver_name!r} at tol={tol:g}: {exc}")
                continue
            rows.append({
                "solver": solver_name, "guess": guess_name, "tolerance": tol,
                "P": fitted.P, "tp": fitted.tp, "e": fitted.e,
                "omega": fitted.omega, "K": fitted.K, "gamma": fitted.gamma,
                "jitter": jitter, "n_solves": n_solves,
                "fit_seconds": timing["fit_seconds"],
                "native_fit_seconds": timing["native_fit_seconds"],
            })
    return pd.DataFrame(rows)


# Fits of the real, unmodified velocities: a solver-agreement check, not an
# orbit measurement (k2-24 and hd164922 are multi-planet systems).
REAL_DATASET_INITIAL_GUESSES: dict[str, OrbitParams] = {
    "k2-24": OrbitParams(P=20.885258, tp=2067.706016317427, e=0.15, omega=0.3, K=3.0, gamma=0.0),
    "hd164922": OrbitParams(P=1207.0, tp=2455474.0, e=0.1, omega=0.0, K=5.0, gamma=0.0),
    "k2-131": OrbitParams(P=0.3693038, tp=2457782.65615, e=0.05, omega=0.0, K=3.0, gamma=0.0),
}

# Transit times from photometry, as in RadVel's example_planets setups
# (K2-131: Dai et al. 2017). HD 164922 b does not transit.
REAL_DATASET_TRANSIT_TC: dict[str, float] = {
    "k2-24": 2072.79438,
    "k2-131": 2457582.9360,
}


def run_real_data_check(dataset_name: str, solvers: list[str] | None = None,
                        tolerances: list[float] | None = None,
                        guess_name: str = "canonical",
                        max_iter: int = 50) -> pd.DataFrame:
    if dataset_name not in REAL_DATASET_INITIAL_GUESSES:
        raise KeyError(
            f"no real-data initial guess registered for {dataset_name!r}; "
            f"known: {sorted(REAL_DATASET_INITIAL_GUESSES)}"
        )
    solvers = solvers if solvers is not None else ["newton", "danby", "markley", "nwm9", "nwm11"]
    tolerances = tolerances if tolerances is not None else [1e-14]
    dataset = load_rv_dataset(dataset_name)
    initial_guess = REAL_DATASET_INITIAL_GUESSES[dataset_name]

    # With P/tc free, sparse real data let P jump to an alias (K2-131 went to
    # 3.02 d) or e run off to ~0.95 (K2-24), so both are fixed here.
    df = run_tolerance_sweep(dataset, initial_guess, solvers, tolerances,
                             guess_name=guess_name, max_iter=max_iter,
                             vary_period=False,
                             fixed_tc=REAL_DATASET_TRANSIT_TC.get(dataset_name))
    df.insert(0, "dataset", dataset_name)
    return df


def run_all_real_data_checks(solvers: list[str] | None = None,
                             tolerances: list[float] | None = None,
                             guess_name: str = "canonical",
                             max_iter: int = 50) -> pd.DataFrame:
    frames = [
        run_real_data_check(name, solvers=solvers, tolerances=tolerances,
                            guess_name=guess_name, max_iter=max_iter)
        for name in REAL_DATASET_INITIAL_GUESSES
    ]
    return pd.concat(frames, ignore_index=True)


def run_reference_mcmc(dataset: pd.DataFrame, initial_guess: OrbitParams,
                       nrun: int | None = None, seed: int = 0) -> dict[str, float]:
    import radvel
    import radvel.fitting

    from keplerbench.rv.radvel_bridge import _find_best_starting_point, build_posterior

    seeded_guess = _find_best_starting_point(dataset, initial_guess)
    post, _, _ = build_posterior(dataset, seeded_guess)
    post = radvel.fitting.maxlike_fitting(post, verbose=False)

    # list_vary_params() returns None in this RadVel version
    n_free = len(post.name_vary_params())
    nwalkers = max(2 * n_free + 2, 10)
    chain = radvel.mcmc(post, nwalkers=nwalkers, nrun=nrun or 2000, serial=True, headless=True)

    sigma: dict[str, float] = {}
    # convert each sample to physical units before taking the std
    if "secosw1" in chain.columns:
        e_samples = chain["secosw1"] ** 2 + chain["sesinw1"] ** 2
        w_samples = np.arctan2(chain["sesinw1"], chain["secosw1"])
        sigma["e"] = float(e_samples.std())
        sigma["omega"] = float(w_samples.std())
    if "logk1" in chain.columns:
        sigma["K"] = float(np.exp(chain["logk1"]).std())

    name_map = {"jit": "jitter", "per1": "P", "tc1": "tc"}
    converted = {"secosw1", "sesinw1", "logk1"}
    for name in post.name_vary_params():
        if name in converted:
            continue
        samples = chain[name]
        if name.startswith("jit"):
            # the chain visits both signs of jit; only |jit| is meaningful
            samples = samples.abs()
        sigma[name_map.get(name, name)] = float(samples.std())
    return sigma


def run(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    dataset_name = cfg.extra.get("dataset") or "k2-24"
    real_dataset = load_rv_dataset(dataset_name)

    truth = _orbit_params_from_extra(cfg, "injected_truth", INJECTED_TRUTH)
    initial_guess = _orbit_params_from_extra(cfg, "initial_guess", INITIAL_GUESS)
    dataset = inject_synthetic_dataset(real_dataset, truth, seed=cfg.seed)

    tolerances = cfg.extra.get(
        "tolerance_sweep", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    reference_tol = cfg.extra.get("reference_tolerance", 1e-14)
    guess_name = cfg.guesses[0] if cfg.guesses else "canonical"

    fits_df = run_tolerance_sweep(
        dataset, initial_guess, cfg.solvers, tolerances,
        guess_name=guess_name, max_iter=cfg.max_iter,
    )
    fits_df.to_csv(results_path("error_propagation", "raw.csv"), index=False)
    if not fits_df.empty:
        fit_timing(fits_df).to_csv(results_path("error_propagation", "timing.csv"), index=False)

    posterior_sigma: dict[str, float] = {}
    if cfg.extra.get("run_mcmc", True):
        posterior_sigma = run_reference_mcmc(
            dataset, initial_guess, nrun=cfg.extra.get("mcmc_steps"), seed=cfg.seed)
        pd.Series(posterior_sigma, name="posterior_sigma").to_csv(
            results_path("error_propagation", "posterior_sigma.csv"))

    if not fits_df.empty:
        shifts = parameter_shift(fits_df, reference_tol=reference_tol)
        if posterior_sigma:
            shifts = shift_in_sigma(shifts, posterior_sigma)
        shifts.to_csv(results_path("error_propagation", "summary.csv"), index=False)

    return fits_df
