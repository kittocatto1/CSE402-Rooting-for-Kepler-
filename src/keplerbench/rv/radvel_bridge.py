"""Swap one of our solvers into RadVel's fitting pipeline.

This is what makes the downstream study meaningful: instead of reimplementing
an orbit fitter, we let RadVel do the fitting and only replace the Kepler
solver at its core.

Owner: Fariha.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def use_solver(solver_name: str, guess_name: str = "canonical",
               tol: float = 1e-14, max_iter: int = 50):
    """Context manager that monkey-patches RadVel's Kepler solver.

    Usage::

        with use_solver("nwm11", "napier", tol=1e-10):
            post = radvel.fitting.maxlike_fitting(post)

    TODO(Fariha):
      1. ``import radvel.kepler`` and find the function the model layer
         actually calls (radvel may use a compiled ``_kepler`` extension -
         check which path is live on your install, and say so in the report,
         because patching the pure-Python one while the Cython one runs would
         silently do nothing).
      2. Save the original, install a replacement that loops over the input
         M array calling our solver, restore the original in the finally
         block.
      3. Our solvers are scalar; radvel passes arrays. The loop will be slow.
         That is acceptable for a tolerance sweep of maximum-likelihood fits,
         but is NOT acceptable for a full MCMC run - budget accordingly, and
         use radvel's own solver for the MCMC that only supplies posterior
         widths.
      4. Add an assertion that the patch actually took effect (e.g. a call
         counter that must be non-zero after the fit). A silently ineffective
         patch would produce results that look fine and mean nothing - this
         is the single biggest risk in the downstream study.
    """
    # TODO(Fariha): implement per the steps above.
    raise NotImplementedError("use_solver: see TODO above")
    yield  # pragma: no cover - keeps the generator shape obvious


def fit_with_solver(data, initial_params, solver_name: str, tol: float):
    """Run a RadVel maximum-likelihood fit using our solver.

    TODO(Fariha): build the radvel Parameters / RVModel / Likelihood /
    Posterior objects, wrap the fit in :func:`use_solver`, and return the
    fitted parameter values plus the number of Kepler solves performed.
    """
    raise NotImplementedError("fit_with_solver: see TODO above")
