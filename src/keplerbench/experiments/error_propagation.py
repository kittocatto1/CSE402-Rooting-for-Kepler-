"""Work Plan step 4 - how solver error reaches the fitted orbital parameters.

The chain the proposal names:  M -> E -> nu -> v_r  -> fitted (P, e, K)

The question: if the Kepler solver is only accurate to some tolerance, how
much does that move the fitted parameters?  And is that shift big or small
compared with RadVel's own MCMC posterior width?

Why the dataset is a real-cadence, injected-orbit dataset and not a direct
fit of K2-24's real velocities
--------------------------------------------------------------------------
A single-planet fit of the *real* K2-24 velocities was tried first (see the
project's dev notes) and consistently drives eccentricity to exactly 0 for
every one of the five solvers, on Powell and Nelder-Mead alike. That is a
real result, not a bug: K2-24 is a two-planet system and this project's RV
model (``rv/model.py``) is deliberately single-planet, so 32 points cannot
separate a real, small eccentricity for planet b from the unmodeled second
planet's signal. But e = 0 collapses Kepler's equation to the trivial
identity nu = M - the Kepler solver stops mattering at all, which would make
this entire study measure nothing (exactly the "near-circular orbit" trap
the team workflow doc warns about).

The fix used here keeps everything else real - K2-24's genuine observation
times and genuine per-point measurement uncertainties - and injects a
clearly-labelled, moderately eccentric synthetic orbit in place of the real
signal (see ``INJECTED_TRUTH`` / ``inject_synthetic_dataset``). This is a
standard injection-recovery design, not a fabricated result: it is reported
as exactly what it is, and it has the added benefit of a known ground truth
to recover, which is a stronger check than "shift relative to the
tightest-tolerance fit" alone.

Owner: Fariha.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from keplerbench.io.config import load_config
from keplerbench.io.results_io import results_path
from keplerbench.rv.anomaly import mean_anomaly, radial_velocity, true_anomaly
from keplerbench.rv.dataset import load_rv_dataset
from keplerbench.rv.model import OrbitParams
from keplerbench.rv.radvel_bridge import fit_with_solver

#: The orbit injected in place of K2-24's real (near-circular, degenerate
#: for a single-planet model) signal. Period is K2-24 b's real, well-known
#: period; tp/e/omega/K are a deliberately-chosen, moderately eccentric
#: synthetic orbit - NOT a claim about the real planet's orbit.
INJECTED_TRUTH = OrbitParams(P=20.885258, tp=2400.0, e=0.35, omega=1.0, K=5.0, gamma=0.0)

#: Deliberately offset from INJECTED_TRUTH so every fit below has to do real
#: work to recover it (an initial guess sitting exactly on the answer would
#: not exercise anything).
INITIAL_GUESS = OrbitParams(P=INJECTED_TRUTH.P, tp=INJECTED_TRUTH.tp,
                            e=0.15, omega=0.3, K=3.0, gamma=0.0)


def _reference_E(e: float, M: float) -> float:
    """Ground-truth eccentric anomaly via ``scipy.optimize.brentq``.

    Independent of the five solvers under test - the same convention the
    rest of the project uses (see ``tests/test_solvers_classical.py``,
    ``reference/mpmath_reference.py``): never validate a solver, or in this
    case generate a synthetic dataset, using one of the solvers being
    studied. f(E) = E - e sin E - M is monotonic increasing for e < 1, so
    any bracket wide enough to contain e*sin(E) (bounded by e < 1) works.
    """
    f = lambda E: E - e * np.sin(E) - M
    return brentq(f, M - 1.2, M + 1.2, xtol=1e-15, rtol=8.881784197001252e-16)


def noiseless_curve(times, truth: OrbitParams) -> np.ndarray:
    """The exact (noise-free) velocity curve for ``truth``, via the
    independent brentq ground truth rather than any of the 5 solvers under
    test. Shared by the tolerance sweep's dataset injection and by
    ``monte_carlo.py``'s noisy realisations, so both use the same reference
    curve.
    """
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
    """Real K2-24 cadence and noise level, with ``truth`` injected in place
    of the real signal. See the module docstring for why."""
    rng = np.random.default_rng(seed)
    errvel = real_dataset["errvel"].to_numpy(dtype=float)
    velocities = noiseless_curve(real_dataset["time"], truth) + rng.normal(0.0, errvel)

    out = real_dataset.copy()
    out["mnvel"] = velocities
    return out


def run_tolerance_sweep(dataset: pd.DataFrame, initial_guess: OrbitParams,
                        solvers: list[str], tolerances: list[float],
                        guess_name: str = "canonical", max_iter: int = 50) -> pd.DataFrame:
    """Fit ``dataset`` with every (solver, tolerance) pair.

    One row per (solver, tolerance) with the fitted P/tp/e/omega/K/gamma and
    the number of Kepler solves performed. A solver/guess combination that
    is not implemented yet is skipped with a warning, not a crash - other
    teammates' solvers may still be in progress (see ``experiments.runner``
    for the same convention).
    """
    rows = []
    for solver_name in solvers:
        for tol in tolerances:
            try:
                fitted, n_solves, jitter = fit_with_solver(
                    dataset, initial_guess, solver_name, tol,
                    guess_name=guess_name, max_iter=max_iter,
                )
            except NotImplementedError as exc:
                warnings.warn(f"skipping {solver_name!r} at tol={tol:g}: {exc}")
                continue
            rows.append({
                "solver": solver_name, "guess": guess_name, "tolerance": tol,
                "P": fitted.P, "tp": fitted.tp, "e": fitted.e,
                "omega": fitted.omega, "K": fitted.K, "gamma": fitted.gamma,
                "jitter": jitter, "n_solves": n_solves,
            })
    return pd.DataFrame(rows)


def run_reference_mcmc(dataset: pd.DataFrame, initial_guess: OrbitParams,
                       nrun: int | None = None, seed: int = 0) -> dict[str, float]:
    """One MCMC run, using RadVel's own (fast, compiled) solver - it only
    supplies posterior widths for context, so there is no need to patch in
    one of our solvers here (team workflow doc, Fariha section, warning 2).
    Uses :func:`keplerbench.rv.radvel_bridge.build_posterior`, so
    multi-instrument datasets get the same CompositeLikelihood handling
    ``fit_with_solver`` uses.
    """
    import radvel
    import radvel.fitting

    from keplerbench.rv.radvel_bridge import build_posterior

    post, _, _ = build_posterior(dataset, initial_guess)
    post = radvel.fitting.maxlike_fitting(post, verbose=False)

    # post.list_vary_params() mutates internal state and returns None in
    # this RadVel version - name_vary_params() is the one that actually
    # returns the list of free-parameter names.
    n_free = len(post.name_vary_params())
    nwalkers = max(2 * n_free + 2, 10)
    chain = radvel.mcmc(post, nwalkers=nwalkers, nrun=nrun or 2000, serial=True, headless=True)

    # RadVel's own parameter names -> the names used everywhere else in this
    # module (OrbitParams fields), so callers never juggle two vocabularies.
    name_map = {"per1": "P", "tp1": "tp", "e1": "e", "w1": "omega",
                "k1": "K", "gamma": "gamma", "jit": "jitter"}
    return {name_map.get(name, name): float(chain[name].std())
            for name in post.name_vary_params()}


def run(config_path: str) -> pd.DataFrame:
    """Entry point used by scripts/run_error_propagation.py.

    Loads the config (Anisa's ``io/config.py``), builds the injected K2-24
    dataset, sweeps every solver over the tolerance ladder, optionally runs
    one reference MCMC for posterior widths, and writes
    ``results/error_propagation/raw.csv`` (+ ``posterior_sigma.csv``).
    """
    cfg = load_config(config_path)
    dataset_name = cfg.extra.get("dataset") or "k2-24"
    real_dataset = load_rv_dataset(dataset_name)
    dataset = inject_synthetic_dataset(real_dataset, INJECTED_TRUTH, seed=cfg.seed)

    tolerances = cfg.extra.get(
        "tolerance_sweep", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    guess_name = cfg.guesses[0] if cfg.guesses else "canonical"

    fits_df = run_tolerance_sweep(
        dataset, INITIAL_GUESS, cfg.solvers, tolerances,
        guess_name=guess_name, max_iter=cfg.max_iter,
    )
    fits_df.to_csv(results_path("error_propagation", "raw.csv"), index=False)

    if cfg.extra.get("run_mcmc", True):
        sigma = run_reference_mcmc(
            dataset, INITIAL_GUESS, nrun=cfg.extra.get("mcmc_steps"), seed=cfg.seed)
        pd.Series(sigma, name="posterior_sigma").to_csv(
            results_path("error_propagation", "posterior_sigma.csv"))

    return fits_df
