"""The anomaly chain:  M -> E -> nu -> v_r  (Proposal Section 4.3).

This is the path along which a solver's error in E travels into the fitted
orbital parameters, so each link is kept as its own testable function.

Owner: Fariha.
"""

from __future__ import annotations

import math


def mean_anomaly(t: float, P: float, tp: float) -> float:
    """M = 2*pi*(t - tp)/P, wrapped into [0, 2*pi).

    Wrapping matters: an unwrapped M of 1e6 radians loses precision in
    sin(M) and would show up as "solver error" that is nothing of the kind.
    """
    M = 2.0 * math.pi * (t - tp) / P
    return M % (2.0 * math.pi)


def true_anomaly(E: float, e: float) -> float:
    """nu from E, numerically stable form.

    Uses the half-angle formula
        nu = 2 * atan2( sqrt(1+e) * sin(E/2), sqrt(1-e) * cos(E/2) )
    rather than the tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2) version, which
    loses the quadrant and blows up near E = pi.

    This is intentionally left unwrapped beyond (-pi, pi]: nu moves through
    an orbit together with E, and near apoapsis (E just past pi) the correct
    nu is also just past pi, not pi minus something. Forcing the output into
    (-pi, pi] would silently reproduce the exact quadrant bug this formula
    exists to avoid (verified against the linearised dnu/dE prediction).
    """
    return 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )


def radial_velocity(nu: float, K: float, e: float, omega: float,
                    gamma: float = 0.0) -> float:
    """v_r = K * (cos(nu + omega) + e*cos(omega)) + gamma."""
    return K * (math.cos(nu + omega) + e * math.cos(omega)) + gamma


def dE_to_dnu(E: float, e: float) -> float:
    """Sensitivity dnu/dE - how much an error in E is amplified into nu.

    Analytic: dnu/dE = sqrt(1 - e^2) / (1 - e*cos E).

    Worth having explicitly: near e -> 1 and E -> 0 this factor gets large,
    which is the analytic reason the pathological corner matters downstream
    and not just for solver robustness. Include a plot of this in the report.
    """
    return math.sqrt(1.0 - e * e) / (1.0 - e * math.cos(E))
