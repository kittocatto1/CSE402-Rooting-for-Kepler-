"""Reading and writing result tables.

House rules (please follow, it keeps the five of us from clobbering each
other):
  * Raw per-solve rows go to ``results/<experiment>/raw.csv``.
  * Per-iteration histories go to ``results/<experiment>/history.csv``.
  * Aggregated / summary tables go to ``results/<experiment>/summary.csv``.
  * Nothing in ``results/`` is committed to git - it is all reproducible.
  * NEVER hand-edit a result file. If a number looks wrong, fix the code.

Owner: Anisa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from keplerbench.core.types import SolveResult

REPO_ROOT = Path(__file__).resolve().parents[3]


def results_path(experiment: str, filename: str) -> Path:
    """Full path to a result file, creating the directory if needed."""
    d = REPO_ROOT / "results" / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def save_results(results: Iterable[SolveResult], experiment: str,
                 filename: str = "raw.csv") -> Path:
    """Write one row per solve.

    TODO(Anisa): build a DataFrame from ``[r.to_row() for r in results]``,
    write it with ``index=False``, and also write a small ``meta.json``
    alongside recording the config fingerprint, git commit and timestamp so
    every table in the report is traceable.
    """
    raise NotImplementedError("save_results: see TODO above")


def save_history(results: Iterable[SolveResult], experiment: str,
                 filename: str = "history.csv") -> Path:
    """Write one row per (solve, iteration) from ``SolveResult.history``.

    TODO(Anisa): flatten history records, keeping solver/guess/e/M as key
    columns so the convergence-order code can group on them.
    """
    raise NotImplementedError("save_history: see TODO above")


def load_results(experiment: str, filename: str = "raw.csv") -> pd.DataFrame:
    """Read a result table back for analysis or plotting."""
    return pd.read_csv(results_path(experiment, filename))
