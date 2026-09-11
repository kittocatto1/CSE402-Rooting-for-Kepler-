"""The (e, M) benchmark grid (Proposal Section 4.2).

Two regions matter and they need different sampling:

  1. The ORDINARY operating region of real radial-velocity fits - moderate
     eccentricity, M spread over the full circle. This is where the average
     cost numbers in the report come from.

  2. The PATHOLOGICAL CORNER - e close to 1 and M close to 0. This is where
     Newton-type iteration on Kepler's equation is known to be fragile, and
     where the robustness metric will actually separate the methods. A
     uniform grid puts almost no points here, so it must be sampled
     deliberately.

Owner: Mahdi.
"""

from __future__ import annotations

import math

import numpy as np


def uniform_grid(n_e: int = 50, n_M: int = 100,
                 e_max: float = 0.99) -> list[tuple[float, float]]:
    """Regular grid: e in [0, e_max], M in [0, pi].

    M is restricted to [0, pi] because Kepler's equation is symmetric about
    M = pi; covering the full circle would only duplicate work.

    Both degenerate endpoints are INCLUDED: e = 0 and M = 0 give E = M
    exactly, so every solver converges there in one step or none.  They are
    kept because they are a cheap correctness check (any solver that misses
    them is broken), and excluded from the headline averages downstream
    instead - see evaluation/robustness.py region splits.
    """
    es = np.linspace(0.0, e_max, n_e)
    Ms = np.linspace(0.0, math.pi, n_M)
    return [(float(e), float(M)) for e in es for M in Ms]


def pathological_grid(n_e: int = 20, n_M: int = 20,
                      e_min: float = 0.9, M_max: float = 0.1
                      ) -> list[tuple[float, float]]:
    """Dense sampling of the hard corner: e -> 1, M -> 0.

    Logarithmic in both (1 - e) and M: (1 - e) from (1 - e_min) down to 1e-4
    and M from M_max down to 1e-6.  Linear spacing here would put nearly
    every point far from the corner, which is the whole thing we are trying
    to resolve.
    """
    one_minus_e = np.logspace(math.log10(1.0 - e_min), -4.0, n_e)
    Ms = np.logspace(math.log10(M_max), -6.0, n_M)
    return [(float(1.0 - u), float(M)) for u in one_minus_e for M in Ms]


def radvel_operating_grid(n_samples: int = 2000, seed: int = 0
                          ) -> list[tuple[float, float]]:
    """(e, M) drawn from the distribution a real RadVel fit actually visits.

    M is uniform on [0, 2*pi): during an MCMC run the orbital phase visited
    is essentially uniform.

    e is drawn from Beta(0.867, 3.03), the eccentricity distribution Kipping
    (2013, MNRAS 434, L51) fitted to the RV exoplanet sample and the prior
    RadVel-style fits commonly adopt.  It concentrates near e ~ 0.1-0.2 with
    a thin tail to high e, which is what real fits see - a uniform e would
    over-weight orbits that barely exist.
    """
    rng = np.random.default_rng(seed)
    Ms = rng.uniform(0.0, 2.0 * math.pi, size=n_samples)
    es = rng.beta(0.867, 3.03, size=n_samples)
    return [(float(e), float(M)) for e, M in zip(es, Ms)]


def build_grid(spec: dict) -> list[tuple[float, float]]:
    """Dispatch on a config's ``grid:`` block.

    Accepted keys::

        type: "uniform" | "pathological" | "radvel" | "combined"
        uniform:      {n_e, n_M, e_max}          # kwargs for uniform_grid
        pathological: {n_e, n_M, e_min, M_max}
        radvel:       {n_samples, seed}

    "combined" concatenates all three and de-duplicates, keeping the order
    the points were generated in so a ``--limit`` smoke run still covers the
    uniform region first.
    """
    builders = {
        "uniform": uniform_grid,
        "pathological": pathological_grid,
        "radvel": radvel_operating_grid,
    }
    kind = spec.get("type", "uniform")
    if kind == "combined":
        names = list(builders)
    elif kind in builders:
        names = [kind]
    else:
        raise ValueError(f"unknown grid type {kind!r}; known: {sorted(builders)} + 'combined'")

    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for name in names:
        for pt in builders[name](**spec.get(name, {})):
            if pt not in seen:
                seen.add(pt)
                points.append(pt)
    return points
