#!/usr/bin/env python3
"""Work Plan step 2. Usage: python scripts/run_verification.py [config]"""
import sys
from keplerbench.experiments import verification

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/verification.yaml"
    verification.run(cfg)
