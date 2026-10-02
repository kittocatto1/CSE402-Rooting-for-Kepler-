#!/usr/bin/env python3
"""Safeguard experiment. Usage: python scripts/run_safeguard.py [config]"""
import sys

from keplerbench.experiments import safeguard

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/safeguard.yaml"
    safeguard.run(cfg)
