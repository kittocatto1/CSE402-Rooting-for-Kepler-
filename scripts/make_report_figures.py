#!/usr/bin/env python3
"""Regenerate every figure in the report from the result files.

Owner: Anisa (integration).

Run this LAST, after the experiments have written results/. It must not
compute any metric itself - it only loads result tables and calls the
plotting functions, so that every figure in the report is reproducible with
one command.

TODO(Anisa): once the plotting functions exist, call them here in report
order and print the path of each figure written. Fail loudly (not silently)
if a required result file is missing - a stale figure in the report is worse
than a missing one.
"""
from keplerbench.plotting.style import use_report_style


def main() -> None:
    use_report_style()
    raise NotImplementedError("make_report_figures: see TODO in the docstring")


if __name__ == "__main__":
    main()
