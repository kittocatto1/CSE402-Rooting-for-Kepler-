from __future__ import annotations

import math


def mean_anomaly(t: float, P: float, tp: float) -> float:
    M = 2.0 * math.pi * (t - tp) / P
    # wrap: sin() of a huge unwrapped M loses precision
    return M % (2.0 * math.pi)


def true_anomaly(E: float, e: float) -> float:
    # half-angle atan2 form keeps the quadrant; tan(nu/2) blows up at E = pi
    return 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )


def radial_velocity(nu: float, K: float, e: float, omega: float,
                    gamma: float = 0.0) -> float:
    return K * (math.cos(nu + omega) + e * math.cos(omega)) + gamma


def dE_to_dnu(E: float, e: float) -> float:
    return math.sqrt(1.0 - e * e) / (1.0 - e * math.cos(E))
