"""YAML experiment configuration.

Everything that defines a run lives in ``configs/*.yaml`` so that a result
file can always be traced back to the exact settings that produced it.  No
experiment should hard-code a grid, a tolerance or a solver list.

Owner: Anisa.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields
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
    #: Repeats for the wall-clock timing loop. Kept in step with
    #: configs/default.yaml - see tests/test_config.py.
    timing_repeats: int = 1000
    #: RNG seed - set for every experiment so runs are reproducible.
    seed: int = 0
    #: Where results are written, relative to the repo root.
    output_dir: str = "results"
    #: Free-form section for experiment-specific settings.
    extra: dict[str, Any] = field(default_factory=dict)


#: Every field above except ``extra``, which is assembled separately.
_KNOWN_KEYS = tuple(f.name for f in fields(ExperimentConfig) if f.name != "extra")


def _field_default(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None                         # ``name``, which has no default


#: The fallback for every optional key, read off the dataclass rather than
#: repeated below. ``configs/default.yaml`` writes the same values down for
#: humans, and tests/test_config.py asserts the two never drift apart.
_DEFAULTS = {f.name: _field_default(f) for f in fields(ExperimentConfig)}


def _require(kind: str, names: Any, known: list[str], path: Path) -> list[str]:
    """Check a list of registry names and return them in canonical spelling.

    A misspelt solver name is otherwise only noticed once the sweep reaches
    it, which on the full grid can be many minutes in.

    Names are lower-cased because the registry keys on the lower-cased name
    but several downstream comparisons are plain string equality - notably
    ``runner.GUESS_INDEPENDENT``, which is ``{"markley"}``. A config saying
    ``solvers: [Markley]`` would resolve through ``get_solver`` and then miss
    that check, so the closed-form solver would be run once per guess and the
    tables would carry four identical rows.
    """
    if isinstance(names, str):
        # `solvers: danby` instead of `solvers: [danby]` - iterating the
        # string would validate it one character at a time.
        raise ValueError(
            f"{path}: '{kind}' must be a list, got the string {names!r}. "
            f"Write it as [{names}]."
        )
    if not isinstance(names, (list, tuple)):
        raise ValueError(f"{path}: '{kind}' must be a list, got {type(names).__name__}")

    out = [str(n).lower() for n in names]
    unknown = [n for n in out if n not in known]
    if unknown:
        raise ValueError(
            f"{path}: unknown {kind} {unknown}; registered {kind} are {known}"
        )
    return out


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML config file into an :class:`ExperimentConfig`.

    Keys the schema does not know about are collected into ``extra`` rather
    than dropped, so an experiment can carry its own settings without this
    module having to learn about them.  Anything the schema *does* know about
    is validated here - a bad solver name or a negative tolerance fails on
    this line, with the file named, instead of deep inside a grid loop.
    """
    # Imported here, not at module scope: the registry imports every solver
    # and guess module, and those import back through the package.
    from keplerbench.core.registry import list_guesses, list_solvers

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(raw).__name__}")

    data = dict(raw)

    # --- extra: the explicit block, plus anything the schema does not know.
    extra = data.pop("extra", None) or {}
    if not isinstance(extra, dict):
        raise ValueError(f"{path}: 'extra' must be a mapping, got {type(extra).__name__}")
    extra = dict(extra)
    leftover = {k: v for k, v in data.items() if k not in _KNOWN_KEYS}
    clashes = sorted(set(leftover) & set(extra))
    if clashes:
        raise ValueError(
            f"{path}: {clashes} appear both at the top level and under 'extra'; "
            "keep each setting in one place so the fingerprint is unambiguous"
        )
    extra.update(leftover)
    for key in leftover:
        data.pop(key)

    # --- the known keys, coerced to the types the pipeline is written against.
    name = str(data.pop("name", path.stem))
    solvers = _require("solvers", data.pop("solvers", []), list_solvers(), path)
    guesses = _require("guesses", data.pop("guesses", []), list_guesses(), path)

    grid = data.pop("grid", None) or {}
    if not isinstance(grid, dict):
        raise ValueError(f"{path}: 'grid' must be a mapping, got {type(grid).__name__}")

    try:
        tol = float(data.pop("tol", _DEFAULTS["tol"]))
        max_iter = int(data.pop("max_iter", _DEFAULTS["max_iter"]))
        timing_repeats = int(data.pop("timing_repeats", _DEFAULTS["timing_repeats"]))
        seed = int(data.pop("seed", _DEFAULTS["seed"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: non-numeric value in a numeric field: {exc}") from None

    record_history = bool(data.pop("record_history", _DEFAULTS["record_history"]))
    use_reference = bool(data.pop("use_reference", _DEFAULTS["use_reference"]))
    output_dir = str(data.pop("output_dir", _DEFAULTS["output_dir"]))

    # --- validation the experiments would otherwise hit much later.
    if not solvers:
        # Every experiment either sweeps this list or takes solvers[0].
        raise ValueError(f"{path}: 'solvers' is empty; name at least one solver")
    if tol < 0:
        raise ValueError(f"{path}: tol must be >= 0 (0 means 'run exactly max_iter "
                         f"iterations'), got {tol!r}")
    if max_iter < 1:
        raise ValueError(f"{path}: max_iter must be >= 1, got {max_iter!r}")
    if timing_repeats < 1:
        raise ValueError(f"{path}: timing_repeats must be >= 1, got {timing_repeats!r}")

    return ExperimentConfig(
        name=name,
        solvers=solvers,
        guesses=guesses,
        grid=grid,
        tol=tol,
        max_iter=max_iter,
        record_history=record_history,
        use_reference=use_reference,
        timing_repeats=timing_repeats,
        seed=seed,
        output_dir=output_dir,
        extra=extra,
    )


def config_fingerprint(cfg: ExperimentConfig) -> str:
    """Short hash of the config, stamped into output filenames.

    This is how we avoid silently overwriting results from a different
    configuration: change any setting and the fingerprint changes with it.
    Two configs that differ only in key order hash the same, because the
    dump is sorted.
    """
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
