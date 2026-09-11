"""Work Plan step 3 - every solver x every guess over the whole (e, M) grid.

This is the main experiment; most of the report's tables and figures come
from the file it writes.

Owner: Mahdi.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    """Entry point used by scripts/run_grid_benchmark.py.

    TODO(Mahdi):
      1. load_config(config_path).
      2. build_grid(cfg.grid) -> list of (e, M).
      3. If cfg.use_reference: get reference roots for the grid
         (reference.reference_roots_grid) - do this ONCE and reuse.
      4. run_sweep(...) with record_history=cfg.record_history.
      5. save_results(...) to results/grid_benchmark/raw.csv and, if history
         was recorded, save_history(...).
      6. Separately, run the timing loop (runner.time_solve) on a SMALL
         representative subset of the grid - timing every grid point is far
         too slow, and wall-clock only needs enough points to compare
         methods. Write it to results/grid_benchmark/timing.csv.
      7. Call evaluation.aggregate to produce summary.csv.

    Warnings from experience:
      * The full grid x 5 solvers x 4 guesses is large. Add a ``--limit``
        option for smoke runs while developing.
      * Keep record_history off for the big run (memory), on for a small
        dedicated sub-grid used by the convergence-order figures.
    """
    raise NotImplementedError("grid_benchmark.run: see TODO above")
