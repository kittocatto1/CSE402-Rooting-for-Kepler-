#!/usr/bin/env python3
"""Refit three real radial-velocity datasets with each solver in turn.

Checks that the five solvers agree on genuine published velocities
(K2-131, HD 164922, K2-24), not only on the injected orbit of the main
propagation study. Writes results/real_data/summary.csv.

Usage: python scripts/run_real_data_checks.py
"""
from keplerbench.experiments.error_propagation import run_all_real_data_checks
from keplerbench.io.results_io import _write_meta, results_path

if __name__ == "__main__":
    df = run_all_real_data_checks()
    path = results_path("real_data", "summary.csv")
    df.to_csv(path, index=False)
    _write_meta(path, "real_data", len(df), None)
    print(df.to_string(index=False))
    print(f"wrote {path}")
