"""Work Plan step 3 - every solver x every guess over the whole (e, M) grid.

This is the main experiment; most of the report's tables and figures come
from the file it writes.

Outputs, all under ``results/grid_benchmark/``:

  ``raw.csv``      one row per (solver, guess, grid point)
  ``history.csv``  per-iteration records, from the small history sub-grid
                   only - recording history over the full grid would cost
                   gigabytes and is not needed
  ``timing.csv``   median wall-clock per solve on a small representative
                   subset; timing every grid point is far too slow and adds
                   nothing, since wall-clock is only ever a cross-check on
                   the machine-independent cost counters
  ``summary.csv``  aggregated table

Owner: Mahdi.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import pandas as pd

from keplerbench.core.registry import get_guess, get_solver
from keplerbench.evaluation import robustness
from keplerbench.experiments.grid import build_grid
from keplerbench.experiments.runner import GUESS_INDEPENDENT, run_sweep, time_solve
from keplerbench.io.config import load_config
from keplerbench.io.results_io import results_path, save_history, save_results

EXPERIMENT = "grid_benchmark"

#: Throwaway experiment name used only by _preflight, never a real result.
PREFLIGHT_EXPERIMENT = "_preflight"


def _preflight(use_reference: bool) -> None:
    """Fail before the sweep, not after it, if a dependency is still a stub.

    The full grid takes a long time. Discovering at the save step that
    ``save_results`` is unwritten would throw all of it away, so every
    collaborator's entry point this function needs is probed up front with
    arguments cheap enough to be harmless.
    """
    missing: list[str] = []

    # Probed into a throwaway experiment directory, removed below, so that a
    # working save_results does not leave an empty file in results/ that
    # looks like a real but failed run.
    probes: list[tuple[str, Callable[[], object]]] = [
        ("io.results_io.save_results",
         lambda: save_results([], PREFLIGHT_EXPERIMENT, "probe.csv")),
    ]
    if use_reference:
        from keplerbench.reference.mpmath_reference import reference_roots_grid
        probes.append(
            ("reference.mpmath_reference.reference_roots_grid",
             lambda: reference_roots_grid([0.1], [0.2])),
        )

    for name, probe in probes:
        try:
            probe()
        except NotImplementedError:
            missing.append(name)
        except Exception:
            # Any other error means the function exists and ran; a genuine
            # problem with it will surface properly at the real call site.
            pass

    # Remove the probe file, then the directory only if it is empty. Never a
    # recursive delete: the path comes from another module's results_path,
    # and an rmtree driven by someone else's path function is one refactor
    # away from erasing a real results directory.
    probe_file = results_path(PREFLIGHT_EXPERIMENT, "probe.csv")
    probe_file.unlink(missing_ok=True)
    try:
        probe_file.parent.rmdir()
    except OSError:
        pass  # not empty, or never created - either way, leave it alone.

    if missing:
        raise NotImplementedError(
            "grid_benchmark cannot run yet - these are still stubs: "
            + ", ".join(missing)
            + ". Run with use_reference=false in the config to skip the "
              "reference-root dependency."
        )


def _timing_points(points: Sequence[tuple[float, float]],
                   n: int) -> list[tuple[float, float]]:
    """An evenly spread subset of the grid, for the wall-clock loop.

    Evenly spread by position rather than random, so the timing subset is
    reproducible without threading the seed through, and so it still spans
    the easy region and the corner in the proportions the grid does.
    """
    if n <= 0 or n >= len(points):
        return list(points)
    stride = len(points) / n
    return [points[int(i * stride)] for i in range(n)]


def _history_subgrid(extra: dict) -> list[tuple[float, float]]:
    """The small (e, M) block that gets run a second time WITH history.

    Separate from the main sweep on purpose: the convergence figures need
    per-iteration records, and the main grid is far too large to keep them
    for.
    """
    spec = extra.get("history_subgrid") or {}
    e_values = spec.get("e") or []
    M_values = spec.get("M") or []
    return [(float(e), float(M)) for e in e_values for M in M_values]


def _run_timing(solver_names: Sequence[str], guess_names: Sequence[str],
                points: Sequence[tuple[float, float]], repeats: int,
                tol: float, max_iter: int) -> pd.DataFrame:
    """Median seconds per solve for each (solver, guess) on each point."""
    rows = []
    for solver_name in solver_names:
        solver = get_solver(solver_name)
        # Closed-form solvers ignore E0, so timing them once per guess would
        # report the same number four times. Mirrors run_sweep.
        labels = ([("n/a", get_guess(guess_names[0]))]
                  if solver_name in GUESS_INDEPENDENT
                  else [(g, get_guess(g)) for g in guess_names])
        for guess_label, guess in labels:
            for e, M in points:
                try:
                    seconds = time_solve(solver, guess, e, M, repeats=repeats,
                                         tol=tol, max_iter=max_iter)
                except NotImplementedError as exc:
                    warnings.warn(
                        f"timing: skipping {solver_name}+{guess_label}: {exc}",
                        stacklevel=2,
                    )
                    break
                rows.append({"solver": solver_name, "guess": guess_label,
                             "e": e, "M": M, "seconds": seconds,
                             "repeats": repeats})
    return pd.DataFrame(rows)


def run(config_path: str, limit: int | None = None) -> None:
    """Entry point used by scripts/run_grid_benchmark.py.

    ``limit`` truncates the grid for a smoke run while developing. The grid
    builder emits the uniform block first, so a small limit still covers the
    ordinary region rather than only the pathological corner.
    """
    cfg = load_config(config_path)
    _preflight(cfg.use_reference)

    points = build_grid(cfg.grid)
    if limit is not None:
        points = points[:limit]
    print(f"grid_benchmark: {len(points)} points, "
          f"{len(cfg.solvers)} solvers, {len(cfg.guesses)} guesses")

    reference_roots = None
    if cfg.use_reference:
        from keplerbench.reference.mpmath_reference import reference_roots_grid
        # NOTE(Mahdi -> Anisa): passed as two PARALLEL sequences, one entry
        # per grid point, not as axes to take a cartesian product of. Our
        # grid is a union of a uniform block, a log corner block and a random
        # RadVel sample, so it is deliberately not a rectangular lattice -
        # a cartesian reading would blow this up into len(e) * len(M) roots.
        reference_roots = reference_roots_grid([e for e, _ in points],
                                               [M for _, M in points])

    results = run_sweep(
        cfg.solvers, cfg.guesses, points,
        tol=cfg.tol, max_iter=cfg.max_iter,
        record_history=False,           # far too much memory for the full grid
        reference_roots=reference_roots,
    )
    raw_path = save_results(results, EXPERIMENT, "raw.csv")
    print(f"  wrote {raw_path} ({len(results)} rows)")

    # --- the small sub-grid that DOES keep history, for Suchi's figures ---
    subgrid = _history_subgrid(cfg.extra)
    if subgrid:
        with_history = run_sweep(
            cfg.solvers, cfg.guesses, subgrid,
            tol=cfg.tol, max_iter=cfg.max_iter,
            record_history=True,
            reference_roots=reference_roots,
            progress_every=0,
        )
        history_path = save_history(with_history, EXPERIMENT, "history.csv")
        print(f"  wrote {history_path} ({len(subgrid)} points with history)")

    # --- wall-clock, on a small representative subset ---
    n_timing = int(cfg.extra.get("timing_points", 0) or 0)
    if n_timing:
        timing = _run_timing(cfg.solvers, cfg.guesses,
                             _timing_points(points, n_timing),
                             repeats=cfg.timing_repeats,
                             tol=cfg.tol, max_iter=cfg.max_iter)
        timing_path = results_path(EXPERIMENT, "timing.csv")
        timing.to_csv(timing_path, index=False)
        print(f"  wrote {timing_path} ({len(timing)} rows)")

    _write_summary(results)


def _write_summary(results) -> None:
    """Aggregate into summary.csv, and never lose the sweep if that fails.

    ``raw.csv`` is the deliverable and has already been written by the time
    this runs; ``summary.csv`` is derived from it and can be rebuilt in
    seconds. Aggregation living in someone else's module must not be able to
    destroy an hour of sweep, so a stub there is a warning, not a crash.
    """
    df = pd.DataFrame([r.to_row() for r in results])
    try:
        from keplerbench.evaluation.aggregate import summarise_grid
        summary = summarise_grid(df)
    except NotImplementedError:
        warnings.warn(
            "evaluation.aggregate.summarise_grid is still a stub; "
            "falling back to the robustness tables for summary.csv",
            stacklevel=2,
        )
        summary = robustness.failure_rates_by_region(df)

    path = results_path(EXPERIMENT, "summary.csv")
    summary.to_csv(path, index=False)
    print(f"  wrote {path} ({len(summary)} rows)")
