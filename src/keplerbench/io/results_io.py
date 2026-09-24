"""Reading and writing result tables.

House rules (please follow, it keeps the five of us from clobbering each
other):
  * Raw per-solve rows go to ``results/<experiment>/raw.csv``.
  * Per-iteration histories go to ``results/<experiment>/history.csv``.
  * Aggregated / summary tables go to ``results/<experiment>/summary.csv``.
  * Nothing in ``results/`` is committed to git - it is all reproducible.
  * NEVER hand-edit a result file. If a number looks wrong, fix the code.

Every table is written with a sidecar ``<name>.meta.json`` recording the
config fingerprint, the git commit and the time, so any number that reaches
the report can be traced back to the run that produced it.

Owner: Anisa.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from keplerbench.core.types import SolveResult
from keplerbench.io.config import ExperimentConfig, config_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Fixed columns of a raw result row, in the order ``SolveResult.to_row``
#: produces them. Any ``cost_*`` column is appended after these, sorted, so
#: two runs with different solvers still line up column-for-column.
RESULT_COLUMNS = ("solver", "guess", "e", "M", "E", "converged", "iterations",
                  "residual", "error", "wall_time", "failure")

#: Fixed columns of a history row. The first four identify the solve, and the
#: convergence-order code groups on exactly those.
HISTORY_COLUMNS = ("solver", "guess", "e", "M",
                   "iteration", "E", "residual", "error", "step")


def results_path(experiment: str, filename: str) -> Path:
    """Full path to a result file, creating the directory if needed."""
    d = REPO_ROOT / "results" / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def _git_state() -> dict[str, object]:
    """Which commit produced this table, and whether the tree was clean.

    Best effort: a missing git, a detached checkout or a tarball download all
    just leave the fields null rather than failing a run that has already
    done the expensive part.
    """
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=REPO_ROOT, timeout=5,
                                  capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_commit": commit,
        # True means uncommitted edits were present, so the commit alone does
        # not identify the code that ran.
        "git_dirty": None if status is None else bool(status),
    }


def meta_path(table: Path) -> Path:
    """The sidecar belonging to a table.

    Public because a caller that cleans up a table it wrote has to remove
    this too - see ``grid_benchmark._preflight``, whose probe file would
    otherwise leave a stray directory behind on every run.
    """
    return table.parent / f"{table.stem}.meta.json"


def _write_meta(table: Path, experiment: str, n_rows: int,
                config: ExperimentConfig | None) -> Path:
    """Write the sidecar describing one table.

    Named after the table (``raw.meta.json``, ``history.meta.json``) rather
    than a single ``meta.json`` per directory, because an experiment writes
    several tables into one directory and they are not always from the same
    moment.
    """
    meta = {
        "experiment": experiment,
        "table": table.name,
        "n_rows": n_rows,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_name": None if config is None else config.name,
        "config_fingerprint": None if config is None else config_fingerprint(config),
        **_git_state(),
    }
    path = meta_path(table)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return path


def _ordered(rows: list[dict], fixed: tuple[str, ...]) -> pd.DataFrame:
    """Rows to a DataFrame with a stable column order.

    An empty run still gets a header, so ``load_results`` can read the file
    back instead of choking on a zero-byte CSV.
    """
    if not rows:
        return pd.DataFrame(columns=list(fixed))
    df = pd.DataFrame(rows)
    present = [c for c in fixed if c in df.columns]
    cost = sorted(c for c in df.columns if c.startswith("cost_"))
    other = [c for c in df.columns if c not in present and c not in cost]
    return df[present + other + cost]


def save_results(results: Iterable[SolveResult], experiment: str,
                 filename: str = "raw.csv", *,
                 config: ExperimentConfig | None = None) -> Path:
    """Write one row per solve, and return the path written.

    Pass ``config`` when the caller has one - it is what puts the fingerprint
    in the sidecar, and so what lets a table in the report be tied back to
    the settings that produced it.
    """
    df = _ordered([r.to_row() for r in results], RESULT_COLUMNS)
    path = results_path(experiment, filename)
    df.to_csv(path, index=False)
    _write_meta(path, experiment, len(df), config)
    return path


def save_history(results: Iterable[SolveResult], experiment: str,
                 filename: str = "history.csv", *,
                 config: ExperimentConfig | None = None) -> Path:
    """Write one row per (solve, iteration) from ``SolveResult.history``.

    solver/guess/e/M are repeated on every row so the convergence-order code
    can group a single solve's residual sequence back out of the flat table.
    """
    rows = []
    for r in results:
        for rec in r.history:
            row = {
                "solver": r.solver, "guess": r.guess, "e": r.e, "M": r.M,
                "iteration": rec.iteration, "E": rec.E,
                "residual": rec.residual, "error": rec.error, "step": rec.step,
            }
            row.update({f"cost_{k}": v for k, v in rec.cost.items()})
            rows.append(row)

    df = _ordered(rows, HISTORY_COLUMNS)
    path = results_path(experiment, filename)
    df.to_csv(path, index=False)
    _write_meta(path, experiment, len(df), config)
    return path


#: Tokens ``load_results`` treats as missing. This is pandas' own default
#: list with ``"n/a"`` taken out, because that is a real label in our tables,
#: not a missing value: the runner writes ``guess == "n/a"`` for the
#: closed-form solvers, which ignore the starting guess. Read with pandas'
#: defaults the label comes back as NaN, and since ``groupby`` drops NaN keys
#: those solvers then vanish from every summary table without a word.
#: ``""`` stays on the list so a genuinely empty cell is still NaN.
NA_TOKENS = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN",
             "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN",
             "None", "nan", "null"]


def load_results(experiment: str, filename: str = "raw.csv") -> pd.DataFrame:
    """Read a result table back for analysis or plotting.

    Round-trips what :func:`save_results` wrote - see :data:`NA_TOKENS` for
    the one place where pandas' defaults would not.
    """
    return pd.read_csv(results_path(experiment, filename),
                       keep_default_na=False, na_values=NA_TOKENS)


def load_meta(experiment: str, filename: str = "raw.csv") -> dict:
    """Read the sidecar for a table, so a figure can name its provenance."""
    return json.loads(meta_path(results_path(experiment, filename)).read_text())
