"""YAML experiment configuration.

Everything that defines a run lives in ``configs/*.yaml`` so that a result
file can always be traced back to the exact settings that produced it.  No
experiment should hard-code a grid, a tolerance or a solver list.

Owner: Anisa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Parsed contents of one config file."""

    name: str
    #: Solver names to run, e.g. ["newton", "danby", "markley", "nwm9", "nwm11"]
    solvers: list[str] = field(default_factory=list)
    #: Guess names to run in front of each solver.
    guesses: list[str] = field(default_factory=list)
    #: Grid specification - see experiments/grid.py for the accepted keys.
    grid: dict[str, Any] = field(default_factory=dict)
    #: Convergence tolerance. Use 0.0 to force a fixed iteration count.
    tol: float = 1e-14
    max_iter: int = 50
    #: Keep the full per-iteration history (needed for convergence order).
    record_history: bool = False
    #: Compare against the mpmath reference root.
    use_reference: bool = True
    #: Repeats for the wall-clock timing loop.
    timing_repeats: int = 1
    #: RNG seed - set for every experiment so runs are reproducible.
    seed: int = 0
    #: Where results are written, relative to the repo root.
    output_dir: str = "results"
    #: Free-form section for experiment-specific settings.
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML config file into an :class:`ExperimentConfig`.

    TODO(Anisa):
      1. Read the YAML into a dict.
      2. Pull out the known keys; everything left over goes into ``extra``.
      3. Validate: solver and guess names must exist in the registry
         (core.registry.list_solvers / list_guesses) - fail here with a clear
         message rather than deep inside the grid loop.
      4. Validate tol >= 0 and max_iter >= 1.
    """
    raise NotImplementedError("load_config: see TODO above")


def config_fingerprint(cfg: ExperimentConfig) -> str:
    """Short hash of the config, stamped into output filenames.

    TODO(Anisa): hash the sorted key/value pairs (json.dumps with
    sort_keys=True, then hashlib.sha1, first 8 hex chars). This is how we
    avoid silently overwriting results from a different configuration.
    """
    raise NotImplementedError("config_fingerprint: see TODO above")
