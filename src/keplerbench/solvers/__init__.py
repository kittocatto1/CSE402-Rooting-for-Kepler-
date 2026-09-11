"""The five candidate root finders (Proposal Section 3.2).

  Category                Method            Order      Owner
  ----------------------  ----------------  ---------  ------
  Baseline                Newton-Raphson    2          Dipit  (worked example)
  Established Kepler      Danby             4          Dipit
  Established Kepler      Markley           closed     Dipit
  With-memory (2024)      NWM9              8.8989     Suchi
  With-memory (2025)      NWM11             10.7446    Suchi
"""

from keplerbench.solvers.newton import NewtonSolver  # noqa: F401
from keplerbench.solvers.danby import DanbySolver  # noqa: F401
from keplerbench.solvers.markley import MarkleySolver  # noqa: F401
from keplerbench.solvers.nwm9 import NWM9Solver  # noqa: F401
from keplerbench.solvers.nwm11 import NWM11Solver  # noqa: F401
