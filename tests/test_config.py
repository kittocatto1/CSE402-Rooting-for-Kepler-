"""Tests for YAML config loading and fingerprinting. Owner: Anisa."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from keplerbench.io.config import (
    ExperimentConfig,
    _field_default,
    config_fingerprint,
    load_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write(tmp_path: Path, text: str, name: str = "t.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# ----------------------------------------------------------------------
# The real configs must all load - this is what stops a typo in someone's
# experiment block from being discovered mid-run.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert cfg.name
    assert cfg.solvers


def test_default_yaml_matches_the_dataclass_defaults():
    """``configs/default.yaml`` records the defaults for humans, but nothing
    loads it, so a value written there is not actually in force anywhere.
    This pins the two together: change one and this test names the other.
    """
    raw = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    checked = 0
    for f in fields(ExperimentConfig):
        if f.name in ("name", "solvers", "guesses") or f.name not in raw:
            continue
        assert raw[f.name] == _field_default(f), (
            f"configs/default.yaml sets {f.name}={raw[f.name]!r} but the "
            f"ExperimentConfig default is {_field_default(f)!r}"
        )
        checked += 1
    assert checked, "nothing was compared - did default.yaml lose its keys?"


# ----------------------------------------------------------------------
# Unknown keys are collected, not dropped.
# ----------------------------------------------------------------------
def test_unknown_top_level_keys_go_into_extra(tmp_path):
    cfg = load_config(_write(tmp_path, "name: x\nsolvers: [danby]\nmystery: 7\n"))
    assert cfg.extra == {"mystery": 7}


def test_explicit_extra_block_is_merged_with_leftovers(tmp_path):
    cfg = load_config(_write(
        tmp_path, "name: x\nsolvers: [danby]\nmystery: 7\nextra: {dataset: k2-24}\n"))
    assert cfg.extra == {"mystery": 7, "dataset": "k2-24"}


def test_a_key_in_both_places_is_an_error(tmp_path):
    """Ambiguous, and two readings of it would share one fingerprint."""
    with pytest.raises(ValueError, match="both at the top level"):
        load_config(_write(
            tmp_path, "name: x\nsolvers: [danby]\ndataset: a\nextra: {dataset: b}\n"))


# ----------------------------------------------------------------------
# Validation happens here, not deep inside a grid loop.
# ----------------------------------------------------------------------
def test_unknown_solver_name_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown solvers"):
        load_config(_write(tmp_path, "name: x\nsolvers: [danby, dnaby]\n"))


def test_unknown_guess_name_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown guesses"):
        load_config(_write(tmp_path, "name: x\nsolvers: [danby]\nguesses: [napir]\n"))


def test_a_bare_string_is_not_accepted_as_a_solver_list(tmp_path):
    """``solvers: danby`` would otherwise be validated one character at a time."""
    with pytest.raises(ValueError, match="must be a list"):
        load_config(_write(tmp_path, "name: x\nsolvers: danby\n"))


def test_registry_names_are_canonicalised_to_lower_case(tmp_path):
    """get_solver is case-insensitive, but runner.GUESS_INDEPENDENT is a
    plain string set - a capitalised name would resolve and then miss that
    check, running the closed-form solver once per guess."""
    from keplerbench.experiments.runner import GUESS_INDEPENDENT

    cfg = load_config(_write(
        tmp_path, "name: x\nsolvers: [Markley]\nguesses: [Napier]\n"))
    assert cfg.solvers == ["markley"]
    assert cfg.guesses == ["napier"]
    assert cfg.solvers[0] in GUESS_INDEPENDENT


@pytest.mark.parametrize("body, match", [
    ("solvers: []\n", "is empty"),
    ("solvers: [danby]\ntol: -1.0e-9\n", "tol must be"),
    ("solvers: [danby]\nmax_iter: 0\n", "max_iter must be"),
    ("solvers: [danby]\ntiming_repeats: 0\n", "timing_repeats must be"),
    ("solvers: [danby]\ntol: tight\n", "non-numeric"),
    ("solvers: [danby]\ngrid: [1, 2]\n", "'grid' must be a mapping"),
    ("solvers: [danby]\nextra: [1, 2]\n", "'extra' must be a mapping"),
])
def test_invalid_settings_are_rejected(tmp_path, body, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, "name: x\n" + body))


def test_tol_zero_is_allowed(tmp_path):
    """verification.yaml uses tol=0 to force exactly max_iter iterations."""
    assert load_config(_write(tmp_path, "name: x\nsolvers: [danby]\ntol: 0.0\n")).tol == 0.0


def test_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_name_falls_back_to_the_file_stem(tmp_path):
    assert load_config(_write(tmp_path, "solvers: [danby]\n", "my_run.yaml")).name == "my_run"


# ----------------------------------------------------------------------
# Fingerprint: the guard against overwriting results from another config.
# ----------------------------------------------------------------------
def test_fingerprint_ignores_key_order(tmp_path):
    a = load_config(_write(tmp_path, "name: x\nsolvers: [danby]\ntol: 1.0e-14\n", "a.yaml"))
    b = load_config(_write(tmp_path, "tol: 1.0e-14\nsolvers: [danby]\nname: x\n", "b.yaml"))
    assert config_fingerprint(a) == config_fingerprint(b)


def test_fingerprint_changes_when_a_setting_changes(tmp_path):
    a = load_config(_write(tmp_path, "name: x\nsolvers: [danby]\ntol: 1.0e-14\n", "a.yaml"))
    b = load_config(_write(tmp_path, "name: x\nsolvers: [danby]\ntol: 1.0e-13\n", "b.yaml"))
    assert config_fingerprint(a) != config_fingerprint(b)


def test_fingerprint_is_short_and_stable(tmp_path):
    cfg = load_config(CONFIGS / "grid_benchmark.yaml")
    assert len(config_fingerprint(cfg)) == 8
    assert config_fingerprint(cfg) == config_fingerprint(cfg)
