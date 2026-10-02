"""The five experiments.

  Verification         -> verification.py        (Suchi)
  Grid benchmark       -> grid_benchmark.py      (Mahdi)
  Safeguard            -> safeguard.py           (Mahdi)
  Error propagation    -> error_propagation.py   (Fariha)
  Monte Carlo          -> monte_carlo.py         (Fariha)

The real-data check, ``run_real_data_check``, lives in error_propagation.py.

Shared by all of them:
  grid.py    the (e, M) sampling            (Mahdi)
  runner.py  the one fair solve pipeline    (Mahdi)
"""
