"""Swap one of our solvers into RadVel's fitting pipeline.

This is what makes the downstream study meaningful: instead of reimplementing
an orbit fitter, we let RadVel do the fitting and only replace the Kepler
solver at its core.

The single biggest risk here (flagged in the team workflow doc): RadVel ships
a compiled Cython/C Kepler solver (``radvel._kepler``) and uses it by default
whenever it is importable.  On this project's machine it IS importable, which
was verified directly::

    >>> from radvel import _kepler   # succeeds here
    >>> import radvel.kepler as k; k.cext
    True

``radvel.model._standard_rv_calc`` calls ``kepler.rv_drive(t, orbel)`` with no
explicit ``use_c_kepler_solver`` argument, so it always takes the compiled
path on this machine regardless of anything done to ``radvel.kepler.kepler``
or ``radvel.orbit.true_anomaly`` - those pure-Python functions are simply
never reached.  Patching them would be a silent no-op.  The fix used below is
to replace ``radvel.kepler.rv_drive`` itself (the one function every code
path actually calls), so the patch is effective no matter which internal path
RadVel would otherwise have taken.

Owner: Fariha.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from keplerbench.rv.model import OrbitParams, rv_curve


@contextmanager
def use_solver(solver_name: str, guess_name: str = "canonical",
               tol: float = 1e-14, max_iter: int = 50):
    """Context manager that monkey-patches RadVel's Kepler solver.

    Usage::

        with use_solver("nwm11", "napier", tol=1e-10) as counter:
            post = radvel.fitting.maxlike_fitting(post)
        assert counter["n_calls"] > 0   # the patch actually fired

    Yields a small dict, ``{"n_calls": <int>}``, counting the number of
    *individual* Kepler-equation solves performed while the patch was
    active (summed over every time RadVel evaluated the model - once per
    optimiser step, times the number of data points). This is the assertion
    the workflow doc calls "the single biggest risk in the downstream
    study": a patch that silently never fires would produce a result that
    looks fine and means nothing.
    """
    import radvel.kepler

    counter = {"n_calls": 0}
    original_rv_drive = radvel.kepler.rv_drive

    def patched_rv_drive(t, orbel, use_c_kepler_solver=None):
        # Signature matches radvel.kepler.rv_drive exactly (including the
        # now-ignored use_c_kepler_solver flag) so it is a drop-in
        # replacement no matter how a caller invokes it.
        per, tp, e, om, k = orbel
        t = np.asarray(t, dtype=float)

        # Mirror RadVel's own defensive clipping in rv_drive so the ONLY
        # behavioural difference introduced here is which Kepler solver
        # computes E - not a change to RadVel's own robustness handling.
        if per < 0:
            per = 1e-4
        if e < 0:
            e = 0.0
        if e > 0.99:
            e = 0.99

        if e == 0.0:
            # Same shortcut RadVel takes: circular orbit needs no Kepler
            # solve at all (nu == M exactly), so nothing is patched here.
            m = 2.0 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
            return k * np.cos(m + om)

        counter["n_calls"] += t.size
        params = OrbitParams(P=per, tp=tp, e=e, omega=om, K=k, gamma=0.0)
        return rv_curve(t, params, solver_name=solver_name,
                        guess_name=guess_name, tol=tol, max_iter=max_iter)

    radvel.kepler.rv_drive = patched_rv_drive
    try:
        yield counter
    finally:
        radvel.kepler.rv_drive = original_rv_drive


def build_posterior(data, initial_params: OrbitParams, vary_period: bool = True,
                    fixed_tc: float | None = None):
    """Build a single-planet RadVel Posterior for ``data``.

    Basis ``"per tc secosw sesinw logk"`` - RadVel's own recommended
    parameterisation (its K2-24 tutorial uses exactly this), NOT the simpler
    "per tp e w k" this project used at first. Two reasons to prefer it:

      1. ``secosw = sqrt(e)*cos(w)``, ``sesinw = sqrt(e)*sin(w)`` stay smooth
         all the way to e = 0, unlike raw ``(e, w)`` - w (the angle to
         periastron) is undefined for a perfect circle, a coordinate
         singularity right at a boundary the fit needs to be able to
         approach freely.
      2. ``tc`` (time of conjunction) is nearly uncorrelated with e/w,
         whereas ``tp`` swings whenever e/w change during the fit.

    ``per`` is free unless ``vary_period=False``; ``tc`` is free unless
    ``fixed_tc`` is given (a transit time measured by photometry). Holding
    them fixed is only valid when they come from an independent
    measurement (transit photometry, as in RadVel's K2-24 tutorial);
    otherwise they are pinned to whatever the initial guess implied - measured: tc pinned ~0.1 d off the injected
    truth, which biased e by ~0.9 sigma - and the proposal (Section 4.3)
    asks for the solver's effect on the fitted P too.

    A real, measured caveat: ``scipy.optimize.minimize(method="Powell")``
    is a direction-set method, so its search directions ARE the basis's own
    coordinate axes - changing basis changes which paths it tries first,
    and it can converge to a noticeably worse local optimum than the old
    basis found from the exact same starting point (measured directly: a
    single-start fit of the injected HD164922 study regressed from
    log-likelihood -646 to -821 under this basis alone). That is why every
    caller goes through :func:`fit_with_solver`'s multi-start search
    (:func:`_find_best_starting_point`) rather than calling
    ``radvel.fitting.maxlike_fitting`` on a single guess directly - with
    that search (and per/tc free), the injected HD164922 study recovers
    e=0.342, omega=0.997, K=5.03 against the injected 0.35 / 1.0 / 5.0,
    all within 0.5 sigma.

    A second, smaller caveat found the same way: right at e=0 itself, the
    map from (secosw, sesinw) to e is locally flat (de/d(secosw) = 2*secosw
    -> 0), so Powell's fixed step size can overshoot a true optimum sitting
    exactly at that boundary even when started there directly. Measured on
    real K2-24 (single-planet model, real 2-planet data - see
    error_propagation.py's module docstring): the old basis finds
    log-likelihood -98.41 at e~0; the best this basis + multi-start search
    reaches is -98.58 (e~0.15) - a small (~0.2), real, and not fully
    eliminated gap. Reported honestly rather than papered over: for K2-24
    specifically, this basis trades a little precision at that boundary for
    the coordinate-singularity fix, and multi-start narrows but does not
    fully close that gap.

    ``e``, ``omega`` and ``K`` on :class:`~keplerbench.rv.model.OrbitParams`
    are converted to/from ``secosw1``/``sesinw1``/``logk1`` at the boundary
    here (via ``radvel.orbit.timeperi_to_timetrans``/``timetrans_to_timeperi``
    for tp<->tc) - nothing outside this module needs to know RadVel's basis
    changed.

    Handles both single- and multi-instrument datasets. One instrument: a
    single ``RVLikelihood`` with an analytically-marginalised gamma
    (vary=False + linear=True - RadVel's own trick, see
    ``Likelihood.residuals()``). Several distinct values in ``data["tel"]``:
    one ``RVLikelihood`` per instrument, each with its own gamma_<tel> /
    jit_<tel>, combined with ``radvel.likelihood.CompositeLikelihood`` - the
    exact pattern RadVel's own multi-instrument tutorials use.

    Returns:
        (post, gamma_params, jit_params) - the Posterior, and the list of
        gamma/jitter parameter names actually used (length 1 for a
        single-instrument dataset, one per instrument otherwise), so a
        caller can read them back without re-deriving the instrument list.
    """
    import radvel
    import radvel.likelihood
    import radvel.orbit
    import radvel.posterior

    e0, w0 = initial_params.e, initial_params.omega
    secosw0 = math.sqrt(e0) * math.cos(w0)
    sesinw0 = math.sqrt(e0) * math.sin(w0)
    logk0 = math.log(initial_params.K)
    tc0 = radvel.orbit.timeperi_to_timetrans(initial_params.tp, initial_params.P, e0, w0)

    # Move tc to the orbit nearest the data: a tc quoted N periods away is
    # correlated with P by a factor N (N ~ 1e5 for tp=2400 vs JD ~2.45e6),
    # which cripples both Powell and MCMC mixing. Same orbit, same model.
    time_base = float(np.median(np.asarray(data["time"], dtype=float)))
    tc0 += round((time_base - tc0) / initial_params.P) * initial_params.P
    if fixed_tc is not None:
        tc0 = fixed_tc

    params = radvel.Parameters(1, basis="per tc secosw sesinw logk")
    params["per1"] = radvel.Parameter(value=initial_params.P, vary=vary_period)
    params["tc1"] = radvel.Parameter(value=tc0, vary=fixed_tc is None)
    params["secosw1"] = radvel.Parameter(value=secosw0)
    params["sesinw1"] = radvel.Parameter(value=sesinw0)
    params["logk1"] = radvel.Parameter(value=logk0)

    model = radvel.RVModel(params, time_base=time_base)

    telescopes = sorted(data["tel"].unique())
    like_list = []
    for tel in telescopes:
        subset = data[data["tel"] == tel]
        suffix = "" if len(telescopes) == 1 else f"_{tel}"
        like = radvel.likelihood.RVLikelihood(
            model, subset["time"], subset["mnvel"], subset["errvel"], suffix=suffix)

        # One real, shared systemic velocity only makes sense for a single
        # instrument; each extra instrument has its own zero-point offset,
        # so seed it from that instrument's own data rather than the
        # (otherwise meaningless) single initial_params.gamma.
        gamma_value = (initial_params.gamma if len(telescopes) == 1
                      else float(np.median(subset["mnvel"])))
        jit_value = float(np.std(subset["errvel"])) or 1.0
        like.params[like.gamma_param] = radvel.Parameter(value=gamma_value, vary=False, linear=True)
        like.params[like.jit_param] = radvel.Parameter(value=jit_value, vary=True)
        like_list.append(like)

    likelihood = like_list[0] if len(like_list) == 1 else radvel.likelihood.CompositeLikelihood(like_list)

    post = radvel.posterior.Posterior(likelihood)
    post.priors += [radvel.EccentricityPrior(1)]
    return post, [l.gamma_param for l in like_list], [l.jit_param for l in like_list]


def _orbit_params_from_post(post, gamma_params: list[str]) -> OrbitParams:
    """Read a fitted ``per tc secosw sesinw logk`` Posterior back into an
    OrbitParams (P, tp, e, omega, K) - the inverse of the conversion
    :func:`build_posterior` does at setup. Shared by :func:`fit_with_solver`
    and ``error_propagation.run_reference_mcmc``.
    """
    import radvel.orbit

    secosw = post.params["secosw1"].value
    sesinw = post.params["sesinw1"].value
    e = secosw**2 + sesinw**2
    omega = math.atan2(sesinw, secosw)
    K = math.exp(post.params["logk1"].value)
    per = post.params["per1"].value
    tc = post.params["tc1"].value
    tp = radvel.orbit.timetrans_to_timeperi(tc, per, e, omega)

    return OrbitParams(
        P=per, tp=tp, e=e, omega=omega, K=K,
        gamma=(post.params[gamma_params[0]].value if len(gamma_params) == 1
              else float("nan")),
    )


#: Coarse (e, omega) grid used to find a good starting basin before the
#: expensive, solver-under-test refinement fit - see _find_best_starting_point.
_E_RESTART_GRID = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7)
_OMEGA_RESTART_GRID = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)


def _find_best_starting_point(data, initial_params: OrbitParams,
                              vary_period: bool = True,
                              fixed_tc: float | None = None) -> OrbitParams:
    """Cheap multi-start basin search, using RadVel's own fast native solver
    (never one of the 5 under test) to find a good (e, omega) basin before
    handing off to the expensive, solver-under-test fit.

    Why this exists: Powell's search directions are the coordinate axes of
    whatever basis it is given (see build_posterior's docstring), so a
    single starting point can converge to a much worse local optimum purely
    by chance of geometry. Trying every combination of _E_RESTART_GRID x
    _OMEGA_RESTART_GRID (up to 30 quick fits) and keeping whichever reaches
    the highest log-likelihood costs a few seconds on a 400-point dataset -
    negligible next to the real fit that follows.

    ``initial_params.omega`` is used as an extra restart if it is not
    already on the grid, so the caller's own guess is never ignored.
    """
    import radvel.fitting
    import radvel.orbit

    omega_seeds = set(_OMEGA_RESTART_GRID) | {initial_params.omega}
    P0 = initial_params.P
    tc_init = fixed_tc if fixed_tc is not None else radvel.orbit.timeperi_to_timetrans(
        initial_params.tp, P0, initial_params.e, initial_params.omega)

    best_logprob = -math.inf
    best_params = initial_params
    for e_seed in _E_RESTART_GRID:
        for w_seed in omega_seeds:
            # Hold the caller's tc (not tp) across seeds, so a seed changes
            # only the orbit's shape, not where in phase the fit starts.
            tp_seed = radvel.orbit.timetrans_to_timeperi(tc_init, P0, e_seed, w_seed)
            seed = replace(initial_params, e=e_seed, omega=w_seed, tp=tp_seed)
            post, gamma_params, _ = build_posterior(data, seed, vary_period, fixed_tc)
            post = radvel.fitting.maxlike_fitting(post, verbose=False)
            logprob = post.logprob()
            if logprob > best_logprob:
                best_logprob = logprob
                best_params = _orbit_params_from_post(post, gamma_params)
    return best_params


def fit_with_solver(data, initial_params: OrbitParams, solver_name: str,
                    tol: float, guess_name: str = "canonical",
                    max_iter: int = 50, verbose: bool = False,
                    vary_period: bool = True, fixed_tc: float | None = None,
                    timing_out: dict[str, float] | None = None):
    """Run a RadVel maximum-likelihood fit using our solver.

    Runs :func:`_find_best_starting_point` first (a cheap multi-start basin
    search with RadVel's own solver - necessary for this basis to actually
    pay off, see build_posterior's docstring), builds the Posterior at that
    improved starting point with :func:`build_posterior`, fits it with
    :func:`radvel.fitting.maxlike_fitting` while :func:`use_solver` is
    active, and returns the fitted parameters plus the number of Kepler
    solves actually performed.

    Args:
        data: DataFrame with columns time, mnvel, errvel, tel (as returned
            by :func:`keplerbench.rv.dataset.load_rv_dataset`).
        initial_params: starting guess for the fit.
        solver_name / guess_name / tol / max_iter: which of our solvers to
            patch RadVel's Kepler equation with, and how tightly to run it -
            this is the factor the error-propagation study sweeps.
        verbose: passed to ``maxlike_fitting`` (RadVel's own progress print).
        vary_period: fit P (default). Pass False to hold P at
            ``initial_params.P`` - for a transiting planet whose period is
            known from photometry; a free P on sparse real data can jump to
            an alias (measured: K2-131 fit at P=3.02 d instead of 0.369 d).
        fixed_tc: hold tc at this measured transit time instead of fitting it.
        timing_out: pass a dict to receive wall-clock seconds (proposal
            Section 4.3, "per full RadVel fit"): ``fit_seconds`` for the fit
            using our solver, and ``native_fit_seconds`` for the identical
            fit (same starting point) using RadVel's own compiled solver.
            Neither includes the multi-start search, which always uses
            RadVel's solver and so says nothing about ours.

    Returns:
        (fitted: OrbitParams, n_solves: int, jitter: float) - n_solves is 0
        only if the orbit's eccentricity converged to exactly 0 (the
        no-Kepler-solve shortcut); a normal fit that never triggers the
        patch at all raises, since that would mean the patch silently did
        not take effect. ``jitter`` is RadVel's fitted excess-noise term
        (|jit| averaged across instruments for a multi-instrument dataset),
        useful as a realistic per-point noise level for Monte Carlo
        realisations downstream. ``fitted.gamma`` is NaN for a
        multi-instrument dataset - there is no single systemic velocity to
        report, only one per instrument.
    """
    import radvel.fitting

    seeded_params = _find_best_starting_point(data, initial_params, vary_period, fixed_tc)
    post, gamma_params, jit_params = build_posterior(data, seeded_params, vary_period, fixed_tc)

    if timing_out is not None:
        native_post, _, _ = build_posterior(data, seeded_params, vary_period, fixed_tc)
        start = time.perf_counter()
        radvel.fitting.maxlike_fitting(native_post, verbose=False)
        timing_out["native_fit_seconds"] = time.perf_counter() - start

    with use_solver(solver_name, guess_name=guess_name, tol=tol,
                    max_iter=max_iter) as counter:
        start = time.perf_counter()
        post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
        fit_seconds = time.perf_counter() - start
        n_solves = counter["n_calls"]
    if timing_out is not None:
        timing_out["fit_seconds"] = fit_seconds

    fitted = _orbit_params_from_post(post, gamma_params)

    if n_solves == 0 and fitted.e != 0.0:
        raise RuntimeError(
            "use_solver patch never fired during the fit (n_calls == 0) but "
            "e != 0 - the patch is not taking effect. Check whether RadVel "
            "picked a different internal code path than radvel.kepler.rv_drive."
        )

    # RadVel's likelihood only uses jit**2, so the fitted sign is arbitrary;
    # averaging signed values across instruments could cancel them out.
    jitter = float(np.mean([abs(post.params[p].value) for p in jit_params]))
    return fitted, n_solves, jitter
