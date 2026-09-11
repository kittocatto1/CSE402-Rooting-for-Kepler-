#!/usr/bin/env python3
"""Work Plan step 5. Usage: python scripts/run_monte_carlo.py [config]"""
import sys
from keplerbench.experiments import monte_carlo

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/monte_carlo.yaml"
    monte_carlo.run(cfg)
