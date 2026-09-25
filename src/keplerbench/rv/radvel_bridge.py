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
    import radvel.kepler

    # Patch rv_drive itself: RadVel calls its compiled C solver from there,
    # so patching radvel.kepler.kepler would silently do nothing.
    counter = {"n_calls": 0}
    original_rv_drive = radvel.kepler.rv_drive

    def patched_rv_drive(t, orbel, use_c_kepler_solver=None):
        per, tp, e, om, k = orbel
        t = np.asarray(t, dtype=float)

        # same clipping as RadVel's rv_drive
        if per < 0:
            per = 1e-4
        if e < 0:
            e = 0.0
        if e > 0.99:
            e = 0.99

        if e == 0.0:
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
    import radvel
    import radvel.likelihood
    import radvel.orbit
    import radvel.posterior

    e0, w0 = initial_params.e, initial_params.omega
    secosw0 = math.sqrt(e0) * math.cos(w0)
    sesinw0 = math.sqrt(e0) * math.sin(w0)
    logk0 = math.log(initial_params.K)
    tc0 = radvel.orbit.timeperi_to_timetrans(initial_params.tp, initial_params.P, e0, w0)

    # Shift tc to the orbit nearest the data; a tc ~1e5 periods away is
    # almost perfectly correlated with P and the MCMC never mixes.
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

        # each instrument has its own zero point
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


# Powell searches along the basis axes and can stall in a poor optimum from a
# single start, so a quick native-solver fit is run from each grid point first.
_E_RESTART_GRID = (0.0, 0.05, 0.15, 0.3, 0.5, 0.7)
_OMEGA_RESTART_GRID = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)


def _find_best_starting_point(data, initial_params: OrbitParams,
                              vary_period: bool = True,
                              fixed_tc: float | None = None) -> OrbitParams:
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
            # keep tc fixed across seeds so only the orbit shape changes
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

    # the likelihood only uses jit**2, so the fitted sign is arbitrary
    jitter = float(np.mean([abs(post.params[p].value) for p in jit_params]))
    return fitted, n_solves, jitter
