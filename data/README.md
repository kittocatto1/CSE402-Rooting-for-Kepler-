# Data

Raw radial-velocity data goes in `data/raw/` and is **git-ignored**.

Nothing here is committed — instead, record below exactly where each file
came from, so anyone on the team can reproduce the download.

## Datasets used

| File | System | Source / citation | Downloaded by | Date |
|------|--------|-------------------|---------------|------|
| _TODO (Fariha)_ | | | | |

Notes for whoever fills this in:

- Pick a system with a **moderately eccentric** orbit. A near-circular orbit
  makes the Kepler solver almost irrelevant, and the error-propagation study
  would measure nothing.
- RadVel ships example datasets; using one of those makes our numbers
  reproducible by anyone who has RadVel installed, and is the lowest-friction
  option.
- Record the instrument(s), the number of measurements and the reported
  uncertainties — the report has to cite all of it.
