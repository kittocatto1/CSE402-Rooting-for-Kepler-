"""Work Plan step 5 - solver-induced shift vs measurement-noise scatter.

Step 4 tells us how much the fitted parameters move when the solver is
sloppy.  On its own that number is meaningless: it only matters relative to
how much the parameters move anyway because the data are noisy.  This
experiment measures that second quantity.

Owner: Fariha.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    """Entry point used by scripts/run_monte_carlo.py.

    TODO(Fariha):
      1. Take the best-fit model from the error-propagation study as truth.
      2. Generate N noisy realisations of the dataset by adding noise drawn
         from the reported per-measurement uncertainties (plus jitter if the
         RadVel fit includes it). Seed the RNG from the config.
      3. Refit each realisation with ONE fixed solver at tight tolerance.
      4. The spread (std dev) of the fitted P, e, K across realisations is
         the measurement-noise-driven uncertainty.
      5. Produce the comparison the proposal asks for, as one table:
             parameter | solver-induced shift | noise-driven spread | MCMC sigma
      6. Write results/monte_carlo/raw.csv and summary.csv.

    N is the cost driver: start with N = 50 to debug the pipeline, then do
    the real run. Record the N actually used in the report.
    """
    raise NotImplementedError("monte_carlo.run: see TODO above")
