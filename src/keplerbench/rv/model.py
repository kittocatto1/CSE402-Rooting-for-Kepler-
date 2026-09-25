from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.runner import solve_one
from keplerbench.rv.anomaly import mean_anomaly, radial_velocity, true_anomaly


@dataclass
class OrbitParams:
    P: float
    tp: float  # time of periastron
    e: float
    omega: float
    K: float
    gamma: float = 0.0


def rv_curve(times, params: OrbitParams, solver_name: str = "danby",
             guess_name: str = "canonical", tol: float = 1e-14,
             max_iter: int = 50,
             cost_out: dict[str, int] | None = None) -> np.ndarray:
    solver = get_solver(solver_name)
    guess = get_guess(guess_name)

    velocities = np.empty(len(times), dtype=float)
    total_cost: dict[str, int] = {}
    for i, t in enumerate(times):
        M = mean_anomaly(t, params.P, params.tp)
        result = solve_one(solver, guess, params.e, M, tol=tol, max_iter=max_iter)
        nu = true_anomaly(result.E, params.e)
        velocities[i] = radial_velocity(nu, params.K, params.e, params.omega,
                                        params.gamma)
        for key, value in result.cost.items():
            total_cost[key] = total_cost.get(key, 0) + value

    if cost_out is not None:
        cost_out.update(total_cost)
    return velocities


def chi_squared(times, velocities, errors, params: OrbitParams, **solver_kw) -> float:
    model = rv_curve(times, params, **solver_kw)
    residual = (np.asarray(velocities, dtype=float) - model) / np.asarray(errors, dtype=float)
    return float(np.sum(residual ** 2))
