"""Work Plan step 4 - how solver error reaches the fitted orbital parameters.

The chain the proposal names:  M -> E -> nu -> v_r  -> fitted (P, e, K)

The question: if the Kepler solver is only accurate to some tolerance, how
much does that move the fitted parameters?  And is that shift big or small
compared with RadVel's own MCMC posterior width?

Why the dataset is a real-cadence, injected-orbit dataset and not a direct
fit of K2-24's real velocities
--------------------------------------------------------------------------
A single-planet fit of the *real* K2-24 velocities was tried first (see
``run_real_data_check("k2-24")``) and lands at a low-to-moderate,
optimiser/starting-point-sensitive eccentricity rather than one clean
answer. Two real, separately-measured reasons contribute:

  1. K2-24 is a two-planet system and this project's RV model
     (``rv/model.py``) is deliberately single-planet, so 32 points cannot
     cleanly separate a real eccentricity for planet b from the unmodeled
     second planet's signal - several different (e, omega, K) combinations
     explain away a similar amount of that contamination almost equally
     well, so the fit has more than one comparably-good local optimum.
  2. Under RadVel's own basis (see ``rv.radvel_bridge.build_posterior``),
     the very small eccentricities this ambiguity favours sit right at a
     coordinate flat-spot (de/d(secosw) -> 0 as e -> 0), which the
     optimiser does not always resolve precisely even with a multi-start
     search - a small, measured, and honestly-reported residual (see that
     module's docstring for the exact numbers).

Either way, a low/near-zero eccentricity outcome would collapse Kepler's
equation to (or near) the trivial identity nu = M - the Kepler solver stops
mattering, which would make this entire study measure nothing (exactly the
"near-circular orbit" trap the team workflow doc warns about).

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

from keplerbench.evaluation.propagation import fit_timing, parameter_shift, shift_in_sigma
from keplerbench.io.config import ExperimentConfig, load_config
from keplerbench.io.results_io import results_path
from keplerbench.rv.anomaly import mean_anomaly, radial_velocity, true_anomaly
from keplerbench.rv.dataset import load_rv_dataset
from keplerbench.rv.model import OrbitParams
from keplerbench.rv.radvel_bridge import fit_with_solver

#: The orbit injected in place of K2-24's real (near-circular, degenerate
#: for a single-planet model) signal. Period is K2-24 b's real, well-known
#: period; tp/e/omega/K are a deliberately-chosen, moderately eccentric
#: synthetic orbit - NOT a claim about the real planet's orbit.
#:
#: Used as the DEFAULT for direct calls (and tests, which call
#: inject_synthetic_dataset/run_tolerance_sweep directly and should not need
#: a config file). run(config_path) instead reads extra.injected_truth /
#: extra.initial_guess when present, so a run's ground truth is traceable to
#: its YAML config rather than hidden in source - see _orbit_params_from_extra.
INJECTED_TRUTH = OrbitParams(P=20.885258, tp=2400.0, e=0.35, omega=1.0, K=5.0, gamma=0.0)

#: Deliberately offset from INJECTED_TRUTH so every fit below has to do real
#: work to recover it (an initial guess sitting exactly on the answer would
#: not exercise anything).
INITIAL_GUESS = OrbitParams(P=INJECTED_TRUTH.P, tp=INJECTED_TRUTH.tp,
                            e=0.15, omega=0.3, K=3.0, gamma=0.0)


def _orbit_params_from_extra(cfg: ExperimentConfig, key: str,
                             default: OrbitParams) -> OrbitParams:
    """Build an OrbitParams from ``cfg.extra[key]`` (a dict with the same
    fields as OrbitParams: P, tp, e, omega, K, gamma) if present, else
    ``default``. Lets ``configs/*.yaml`` override the ground truth / initial
    guess a run actually used, instead of it only being visible in source -
    ``io/config.py``'s own rule ("no experiment should hard-code a grid, a
    tolerance or a solver list") applies just as much to the truth an
    injection-recovery study is tested against.
    """
    spec = cfg.extra.get(key)
    return default if spec is None else OrbitParams(**spec)


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
                        guess_name: str = "canonical", max_iter: int = 50,
                        vary_period: bool = True,
                        fixed_tc: float | None = None) -> pd.DataFrame:
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


#: Real-data validation cases, NOT the main study - direct fits of each
#: dataset's REAL, unmodified velocities, no injection. Only k2-131 is
#: literature-comparable: it is a genuine SINGLE-planet system, so there is
#: no unmodeled second planet to blame a result on (P = 0.3693038 d, e ~ 0 -
#: NASA Exoplanet Archive, tidally circularised). k2-24 and hd164922 are
#: real multi-planet systems - a single-planet fit here is exploratory only
#: (exactly the degeneracy the injection-recovery study above exists to
#: work around), not expected to match any published orbit. All three exist
#: to show the pipeline (dataset loading, multi-instrument
#: CompositeLikelihood, the RadVel basis + multi-start search, all 5
#: solvers) behaves sensibly on completely real data.
REAL_DATASET_INITIAL_GUESSES: dict[str, OrbitParams] = {
    "k2-24": OrbitParams(P=20.885258, tp=2067.706016317427, e=0.15, omega=0.3, K=3.0, gamma=0.0),
    "hd164922": OrbitParams(P=1207.0, tp=2455474.0, e=0.1, omega=0.0, K=5.0, gamma=0.0),
    "k2-131": OrbitParams(P=0.3693038, tp=2457782.65615, e=0.05, omega=0.0, K=3.0, gamma=0.0),
}

#: Transit times measured by photometry, held fixed in the real-data checks
#: exactly as RadVel's own example setups do (example_planets/
#: epic203771098.py: tc1=2072.79438, vary=False; example_planets/k2-131.py:
#: Tc=2457582.9360 +/- 0.0011, Dai et al. 2017). HD 164922 b does not
#: transit, so its tc is fitted.
REAL_DATASET_TRANSIT_TC: dict[str, float] = {
    "k2-24": 2072.79438,
    "k2-131": 2457582.9360,
}


def run_real_data_check(dataset_name: str, solvers: list[str] | None = None,
                        tolerances: list[float] | None = None,
                        guess_name: str = "canonical",
                        max_iter: int = 50) -> pd.DataFrame:
    """Direct fit of ``dataset_name``'s REAL, unmodified velocities - no
    injection. See :data:`REAL_DATASET_INITIAL_GUESSES` for which dataset
    (only k2-131) is literature-comparable and which are exploratory-only.

    Not part of the main injection-recovery study above - for k2-24/
    hd164922, a genuinely circular-ish real orbit or an unmodeled extra
    planet makes the Kepler solver's hard case less exercised here too
    (see the module docstring), just for real astrophysical/model reasons
    rather than the coordinate-singularity issue the injection design works
    around.
    """
    if dataset_name not in REAL_DATASET_INITIAL_GUESSES:
        raise KeyError(
            f"no real-data initial guess registered for {dataset_name!r}; "
            f"known: {sorted(REAL_DATASET_INITIAL_GUESSES)}"
        )
    solvers = solvers if solvers is not None else ["newton", "danby", "markley", "nwm9", "nwm11"]
    tolerances = tolerances if tolerances is not None else [1e-14]
    dataset = load_rv_dataset(dataset_name)
    initial_guess = REAL_DATASET_INITIAL_GUESSES[dataset_name]

    # P held at the registered (published) period, and tc at the measured
    # transit time where there is one: with them free, sparse real data plus
    # a single-planet model lets P jump to an alias (measured: K2-131 at
    # P=3.02 d) or e run off to ~0.95 (K2-24).
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
    """:func:`run_real_data_check` on every registered real dataset
    (k2-24, hd164922, k2-131), concatenated into one DataFrame tagged by
    the ``dataset`` column."""
    frames = [
        run_real_data_check(name, solvers=solvers, tolerances=tolerances,
                            guess_name=guess_name, max_iter=max_iter)
        for name in REAL_DATASET_INITIAL_GUESSES
    ]
    return pd.concat(frames, ignore_index=True)


def run_reference_mcmc(dataset: pd.DataFrame, initial_guess: OrbitParams,
                       nrun: int | None = None, seed: int = 0) -> dict[str, float]:
    """One MCMC run, using RadVel's own (fast, compiled) solver - it only
    supplies posterior widths for context, so there is no need to patch in
    one of our solvers here (team workflow doc, Fariha section, warning 2).
    Uses :func:`keplerbench.rv.radvel_bridge.build_posterior`, so
    multi-instrument datasets get the same CompositeLikelihood handling
    ``fit_with_solver`` uses.

    ``build_posterior`` fits in RadVel's ``secosw``/``sesinw``/``logk``
    basis, not raw e/omega/K, so the chain's own columns are in that space
    too. std(secosw) is not std(e) - e = secosw**2 + sesinw**2 is a
    nonlinear transform - so every sample is converted to physical units
    first, then the std is taken of THAT, not the other way round.

    Also runs :func:`keplerbench.rv.radvel_bridge._find_best_starting_point`
    first, same as ``fit_with_solver`` - starting the maximum-a-posteriori
    fit (that seeds the walkers) from a poor local optimum would give MCMC
    a bad starting cloud to walk from too.
    """
    import radvel
    import radvel.fitting

    from keplerbench.rv.radvel_bridge import _find_best_starting_point, build_posterior

    seeded_guess = _find_best_starting_point(dataset, initial_guess)
    post, _, _ = build_posterior(dataset, seeded_guess)
    post = radvel.fitting.maxlike_fitting(post, verbose=False)

    # post.list_vary_params() mutates internal state and returns None in
    # this RadVel version - name_vary_params() is the one that actually
    # returns the list of free-parameter names.
    n_free = len(post.name_vary_params())
    nwalkers = max(2 * n_free + 2, 10)
    chain = radvel.mcmc(post, nwalkers=nwalkers, nrun=nrun or 2000, serial=True, headless=True)

    sigma: dict[str, float] = {}
    if "secosw1" in chain.columns:
        e_samples = chain["secosw1"] ** 2 + chain["sesinw1"] ** 2
        w_samples = np.arctan2(chain["sesinw1"], chain["secosw1"])
        sigma["e"] = float(e_samples.std())
        sigma["omega"] = float(w_samples.std())
    if "logk1" in chain.columns:
        sigma["K"] = float(np.exp(chain["logk1"]).std())

    # Everything else (jit, jit_<tel>, ...) is already in physical units -
    # RadVel's own parameter names -> the names used everywhere else in this
    # module (OrbitParams fields), so callers never juggle two vocabularies.
    name_map = {"jit": "jitter", "per1": "P", "tc1": "tc"}
    converted = {"secosw1", "sesinw1", "logk1"}
    for name in post.name_vary_params():
        if name in converted:
            continue
        samples = chain[name]
        if name.startswith("jit"):
            # Likelihood depends on jit**2 only: the chain wanders across both
            # signs, so the std of signed samples overstates the width.
            samples = samples.abs()
        sigma[name_map.get(name, name)] = float(samples.std())
    return sigma


def run(config_path: str) -> pd.DataFrame:
    """Entry point used by scripts/run_error_propagation.py.

    Loads the config (Anisa's ``io/config.py``), builds the injected dataset
    (ground truth from ``extra.injected_truth`` if given, else
    ``INJECTED_TRUTH``), sweeps every solver over the tolerance ladder,
    optionally runs one reference MCMC for posterior widths, and writes:
      - ``results/error_propagation/raw.csv``       (one row per solve)
      - ``results/error_propagation/posterior_sigma.csv``  (if run_mcmc)
      - ``results/error_propagation/timing.csv``    (wall-clock per full
        RadVel fit, per solver - see ``evaluation.propagation.fit_timing``)
      - ``results/error_propagation/summary.csv``   (parameter_shift,
        in units of MCMC sigma when posterior_sigma was computed) - the
        house rule in ``io/results_io.py`` ("Aggregated / summary tables go
        to results/<experiment>/summary.csv") applies here like every other
        experiment.
    """
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
