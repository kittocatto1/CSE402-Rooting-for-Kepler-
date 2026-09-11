"""The anomaly chain:  M -> E -> nu -> v_r  (Proposal Section 4.3).

This is the path along which a solver's error in E travels into the fitted
orbital parameters, so each link is kept as its own testable function.

Owner: Fariha.
"""

from __future__ import annotations


def mean_anomaly(t: float, P: float, tp: float) -> float:
    """M = 2*pi*(t - tp)/P, wrapped into [0, 2*pi).

    TODO(Fariha): implement, including the wrap. Wrapping matters: an
    unwrapped M of 1e6 radians loses precision in sin(M) and would show up
    as "solver error" that is nothing of the kind.
    """
    raise NotImplementedError("mean_anomaly: see TODO above")


def true_anomaly(E: float, e: float) -> float:
    """nu from E, numerically stable form.

    Use the half-angle formula
        nu = 2 * atan2( sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2) )
    rather than the tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2) version, which
    loses the quadrant and blows up near E = pi.

    TODO(Fariha): implement and unit-test the quadrant behaviour across the
    full circle.
    """
    raise NotImplementedError("true_anomaly: see TODO above")


def radial_velocity(nu: float, K: float, e: float, omega: float,
                    gamma: float = 0.0) -> float:
    """v_r = K * (cos(nu + omega) + e*cos(omega)) + gamma.

    TODO(Fariha): implement. Confirm the sign convention matches RadVel's
    before running anything downstream - a sign flip in omega will look like
    a solver problem when it is not.
    """
    raise NotImplementedError("radial_velocity: see TODO above")


def dE_to_dnu(E: float, e: float) -> float:
    """Sensitivity dnu/dE - how much an error in E is amplified into nu.

    Analytic: dnu/dE = sqrt(1 - e^2) / (1 - e*cos E).

    Worth having explicitly: near e -> 1 and E -> 0 this factor gets large,
    which is the analytic reason the pathological corner matters downstream
    and not just for solver robustness. Include a plot of this in the report.

    TODO(Fariha): implement and use it to sanity-check the measured
    propagation - the empirical amplification should track this curve.
    """
    raise NotImplementedError("dE_to_dnu: see TODO above")
