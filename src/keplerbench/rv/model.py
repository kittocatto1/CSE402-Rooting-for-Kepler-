"""A single-planet Keplerian RV model built on our own solvers.

Owner: Fariha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.runner import solve_one
from keplerbench.rv.anomaly import mean_anomaly, radial_velocity, true_anomaly


@dataclass
class OrbitParams:
    """The parameters a single-planet RV fit recovers."""

    P: float       # orbital period
    tp: float      # time of periastron
    e: float       # eccentricity
    omega: float   # argument of periastron
    K: float       # velocity semi-amplitude
    gamma: float = 0.0  # systemic velocity offset


def rv_curve(times, params: OrbitParams, solver_name: str = "danby",
             guess_name: str = "canonical", tol: float = 1e-14,
             cost_out: dict[str, int] | None = None) -> np.ndarray:
    """Model radial velocities at ``times`` using OUR solver, not RadVel's.

    Goes through ``experiments.runner.solve_one`` for every point, so this
    curve is subject to the exact same cost accounting and stopping rule as
    the rest of the benchmark - no shortcuts.  Pass ``cost_out`` (a fresh
    dict) to also collect the total solver cost for the whole curve - "cost
    per full RV fit" is one of the metrics in Section 4.3.
    """
    solver = get_solver(solver_name)
    guess = get_guess(guess_name)

    velocities = np.empty(len(times), dtype=float)
    total_cost: dict[str, int] = {}
    for i, t in enumerate(times):
        M = mean_anomaly(t, params.P, params.tp)
        result = solve_one(solver, guess, params.e, M, tol=tol)
        nu = true_anomaly(result.E, params.e)
        velocities[i] = radial_velocity(nu, params.K, params.e, params.omega,
                                        params.gamma)
        for key, value in result.cost.items():
            total_cost[key] = total_cost.get(key, 0) + value

    if cost_out is not None:
        cost_out.update(total_cost)
    return velocities


def chi_squared(times, velocities, errors, params: OrbitParams, **solver_kw) -> float:
    """Standard chi^2 of the model against the data.

    Used by the maximum-likelihood fits in the error-propagation study.
    """
    model = rv_curve(times, params, **solver_kw)
    residual = (np.asarray(velocities, dtype=float) - model) / np.asarray(errors, dtype=float)
    return float(np.sum(residual ** 2))
