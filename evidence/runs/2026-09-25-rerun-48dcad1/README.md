# Experiment evidence — 25 September 2026

This snapshot preserves the latest rerun behind [the report](../../../report.pdf). The experiments used the implementation at `48dcad1feea73a1d50268e31bf7d40b26ca3a2bc` (Python 3.12.6). The report and figure-layout corrections are part of the commit containing this snapshot.

- `results/verification/`: convergence-order and correctness tables, with provenance sidecars.
- `results/grid_benchmark/`: 125,800 instrumented solves, history and timed subset. The raw and history tables have provenance sidecars; the derived summary and timing tables do not.
- `results/error_propagation/`: solver-tolerance sweep, fit timing and MCMC width for the injected orbit.
- `results/monte_carlo/`: 50 repeated-noise fits and summary.
- `real_data_checks.csv`: 15 direct fits (five solvers on each of K2-24, HD 164922 and K2-131), run separately at tolerance `1e-14` with the canonical start.
- `figures/`: all 12 regenerated report figures in both PDF and PNG form.

The root-level `report.pdf` was rebuilt from `report.tex` and these figures, then visually checked page by page. The final LaTeX build had no warnings; 24 focused plotting tests passed. The full 264-test suite passed earlier in the rerun, before these figure-layout-only edits.

The working `results/` and `figures/` directories remain ignored by Git. This curated copy is the committed proof. The older `results/reference_roots.csv` (dated 24 September) is excluded because it was **not** regenerated in this run. Raw input datasets, source-paper PDFs, virtual environments and TeX build files are also excluded. Some derived/downstream CSVs lack provenance sidecars; their association with this run is documented here, but not independently recorded by those files.
