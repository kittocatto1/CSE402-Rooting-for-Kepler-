"""A single-planet Keplerian RV model built on our own solvers.

Owner: Fariha.
"""

from __future__ import annotations

from dataclasses import dataclass


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
             guess_name: str = "canonical", tol: float = 1e-14):
    """Model radial velocities at ``times`` using OUR solver, not RadVel's.

    TODO(Fariha):
      1. For each time: M = mean_anomaly(t, P, tp).
      2. Solve Kepler with the named solver/guess via
         experiments.runner.solve_one - going through the shared pipeline
         keeps the cost accounting valid here too.
      3. nu = true_anomaly(E, e); v = radial_velocity(nu, K, e, omega, gamma).
      4. Return a numpy array.
      5. Record total solver cost for the whole curve - "cost per full RV
         fit" is one of the metrics in Section 4.3.
    """
    raise NotImplementedError("rv_curve: see TODO above")


def chi_squared(times, velocities, errors, params: OrbitParams, **solver_kw) -> float:
    """Standard chi^2 of the model against the data.

    TODO(Fariha): sum(((v_obs - v_model) / sigma)**2). Used by the
    maximum-likelihood fits in the error-propagation study.
    """
    raise NotImplementedError("chi_squared: see TODO above")
