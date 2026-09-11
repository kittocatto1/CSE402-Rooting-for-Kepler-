# Data

Raw radial-velocity data goes in `data/raw/` and is **git-ignored**.

Nothing here is committed — instead, record below exactly where each file
came from, so anyone on the team can reproduce the download.

## Datasets used

| File | System | Source / citation | Downloaded by | Date |
|------|--------|-------------------|---------------|------|
| `k2-24.csv` (renamed from `epic203771098.csv`) | K2-24 (EPIC 203771098), a two-planet sub-Saturn system | 32 HIRES RV points (columns `t`, `vel`, `errvel`). This is RadVel's own example dataset for its ["K2-24 Fitting+MCMC" tutorial](https://radvel.readthedocs.io/en/latest/tutorials/K2-24_Fitting%2BMCMC.html) — the exact one named in our proposal. Source file: [`example_data/epic203771098.csv`](https://github.com/California-Planet-Search/radvel/blob/master/example_data/epic203771098.csv) in the RadVel GitHub repo. System discovered by Petigura et al. 2016 (AJ 152, 28); RV mass measurements in Petigura et al. 2018 (AJ 155, 21) — cite both for the system itself. | Fariha | 2026-09-11 |

Notes for whoever fills this in:

- Pick a system with a **moderately eccentric** orbit. A near-circular orbit
  makes the Kepler solver almost irrelevant, and the error-propagation study
  would measure nothing.
- RadVel ships example datasets; using one of those makes our numbers
  reproducible by anyone who has RadVel installed, and is the lowest-friction
  option.
- Record the instrument(s), the number of measurements and the reported
  uncertainties — the report has to cite all of it.
