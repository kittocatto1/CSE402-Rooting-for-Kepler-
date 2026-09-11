#!/usr/bin/env python3
"""Work Plan step 4. Usage: python scripts/run_error_propagation.py [config]"""
import sys
from keplerbench.experiments import error_propagation

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/error_propagation.yaml"
    error_propagation.run(cfg)
