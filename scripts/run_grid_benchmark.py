#!/usr/bin/env python3
"""Work Plan step 3. Usage: python scripts/run_grid_benchmark.py [config]"""
import sys
from keplerbench.experiments import grid_benchmark

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/grid_benchmark.yaml"
    grid_benchmark.run(cfg)
