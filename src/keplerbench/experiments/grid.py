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

import numpy as np


def uniform_grid(n_e: int = 50, n_M: int = 100,
                 e_max: float = 0.99) -> list[tuple[float, float]]:
    """Regular grid: e in [0, e_max], M in [0, pi].

    M is restricted to [0, pi] because Kepler's equation is symmetric about
    M = pi; covering the full circle would only duplicate work.

    TODO(Mahdi): build with np.linspace and return a list of (e, M) tuples.
    Decide and document whether the endpoints e = 0 and M = 0 are included -
    both are degenerate (E = M exactly) and can flatter a solver.
    """
    raise NotImplementedError("uniform_grid: see TODO above")


def pathological_grid(n_e: int = 20, n_M: int = 20,
                      e_min: float = 0.9, M_max: float = 0.1
                      ) -> list[tuple[float, float]]:
    """Dense sampling of the hard corner: e -> 1, M -> 0.

    TODO(Mahdi): use LOGARITHMIC spacing in (1 - e) and in M, e.g.
    1 - e from 1e-1 down to 1e-4 and M from 1e-1 down to 1e-6. Linear
    spacing here wastes almost all the points far from the corner.
    """
    raise NotImplementedError("pathological_grid: see TODO above")


def radvel_operating_grid(n_samples: int = 2000, seed: int = 0
                          ) -> list[tuple[float, float]]:
    """(e, M) drawn from the distribution a real RadVel fit actually visits.

    TODO(Mahdi):
      1. Sample M uniform on [0, 2*pi) - during an MCMC run, phase is
         essentially uniform.
      2. Sample e from the eccentricity prior / observed distribution used
         for RV planets rather than uniform on [0, 1). Write down in this
         docstring which distribution you used and why (a Beta distribution
         fitted to known RV eccentricities is the usual choice).
      3. Seed the RNG from the argument so the grid is reproducible.
    """
    raise NotImplementedError("radvel_operating_grid: see TODO above")


def build_grid(spec: dict) -> list[tuple[float, float]]:
    """Dispatch on a config's ``grid:`` block.

    Accepted keys (extend as needed, and document here):
        type: "uniform" | "pathological" | "radvel" | "combined"
        plus whatever that grid function takes.

    TODO(Mahdi): implement the dispatch; for "combined", concatenate the
    three grids and de-duplicate.
    """
    raise NotImplementedError("build_grid: see TODO above")
