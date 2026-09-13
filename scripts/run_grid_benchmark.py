#!/usr/bin/env python3
"""Work Plan step 3.

Usage: python scripts/run_grid_benchmark.py [config] [--limit N]

``--limit`` truncates the grid for a smoke run. The grid builder emits the
uniform block first, so a small limit still covers the ordinary region.
"""
import argparse
import sys

from keplerbench.experiments import grid_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?",
                        default="configs/grid_benchmark.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N grid points (smoke run)")
    args = parser.parse_args()
    try:
        grid_benchmark.run(args.config, limit=args.limit)
    except NotImplementedError as exc:
        # While the five tracks are in flight, "waiting on someone" is a
        # normal state for this script, not a crash. Report it as a status.
        sys.exit(f"{exc}")
