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

from contextlib import contextmanager

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


def build_posterior(data, initial_params: OrbitParams):
    """Build a single-planet RadVel Posterior for ``data``.

    Basis ``"per tp e w k"`` maps directly onto
    :class:`~keplerbench.rv.model.OrbitParams` with no secosw/sesinw
    conversion. Period and time of periastron are held fixed at their
    (externally well-determined, e.g. transit-photometry) starting values -
    the same thing RadVel's own K2-24 tutorial does for per/tc. Refitting
    them from RV data alone on a short, sparse dataset is an aliasing trap
    unrelated to what this study measures; e, omega, K and every
    gamma/jitter are left free.

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
    import radvel.posterior

    params = radvel.Parameters(1, basis="per tp e w k")
    params["per1"] = radvel.Parameter(value=initial_params.P, vary=False)
    params["tp1"] = radvel.Parameter(value=initial_params.tp, vary=False)
    params["e1"] = radvel.Parameter(value=initial_params.e)
    params["w1"] = radvel.Parameter(value=initial_params.omega)
    params["k1"] = radvel.Parameter(value=initial_params.K)

    time_base = float(np.median(np.asarray(data["time"], dtype=float)))
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


def fit_with_solver(data, initial_params: OrbitParams, solver_name: str,
                    tol: float, guess_name: str = "canonical",
                    max_iter: int = 50, verbose: bool = False):
    """Run a RadVel maximum-likelihood fit using our solver.

    Builds the Posterior with :func:`build_posterior`, fits it with
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

    Returns:
        (fitted: OrbitParams, n_solves: int, jitter: float) - n_solves is 0
        only if the orbit's eccentricity converged to exactly 0 (the
        no-Kepler-solve shortcut); a normal fit that never triggers the
        patch at all raises, since that would mean the patch silently did
        not take effect. ``jitter`` is RadVel's fitted excess-noise term
        (averaged across instruments for a multi-instrument dataset),
        useful as a realistic per-point noise level for Monte Carlo
        realisations downstream. ``fitted.gamma`` is NaN for a
        multi-instrument dataset - there is no single systemic velocity to
        report, only one per instrument.
    """
    import radvel.fitting

    post, gamma_params, jit_params = build_posterior(data, initial_params)

    with use_solver(solver_name, guess_name=guess_name, tol=tol,
                    max_iter=max_iter) as counter:
        post = radvel.fitting.maxlike_fitting(post, verbose=verbose)
        n_solves = counter["n_calls"]

    if n_solves == 0 and post.params["e1"].value != 0.0:
        raise RuntimeError(
            "use_solver patch never fired during the fit (n_calls == 0) but "
            "e != 0 - the patch is not taking effect. Check whether RadVel "
            "picked a different internal code path than radvel.kepler.rv_drive."
        )

    fitted = OrbitParams(
        P=post.params["per1"].value,
        tp=post.params["tp1"].value,
        e=post.params["e1"].value,
        omega=post.params["w1"].value,
        K=post.params["k1"].value,
        gamma=(post.params[gamma_params[0]].value if len(gamma_params) == 1
              else float("nan")),
    )
    jitter = float(np.mean([post.params[p].value for p in jit_params]))
    return fitted, n_solves, jitter
