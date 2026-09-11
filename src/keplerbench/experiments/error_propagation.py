"""Work Plan step 4 - how solver error reaches the fitted orbital parameters.

The chain the proposal names:  M -> E -> nu -> v_r  -> fitted (P, e, K)

The question: if the Kepler solver is only accurate to some tolerance, how
much does that move the fitted parameters?  And is that shift big or small
compared with RadVel's own MCMC posterior width?

Owner: Fariha.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    """Entry point used by scripts/run_error_propagation.py.

    TODO(Fariha):
      1. Load a REAL radial-velocity dataset (rv/dataset.py). Record which
         system and which published dataset - the report must name it.
      2. For each solver and each tolerance in a sweep
         (e.g. 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
           a. Fit the orbit with that solver at that tolerance
              (rv/radvel_bridge.py swaps our solver into RadVel).
           b. Record the best-fit P, e, K (and omega, tp).
      3. Take the tightest tolerance as the reference fit, and report the
         SHIFT in each parameter as tolerance is loosened.
      4. Express every shift in units of the MCMC posterior sigma for that
         parameter, so "does this matter?" has a quantitative answer.
      5. Write results/error_propagation/raw.csv. Do not fabricate any of
         these numbers - if a fit does not converge, record it as a failure.

    Design note: keep the MCMC run separate and cached. Re-running MCMC for
    every tolerance is wasteful; one MCMC run gives the posterior widths, and
    the tolerance sweep only needs maximum-likelihood fits.
    """
    raise NotImplementedError("error_propagation.run: see TODO above")
