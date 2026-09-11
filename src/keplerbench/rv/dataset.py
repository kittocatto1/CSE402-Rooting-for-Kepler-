"""Loading the real radial-velocity dataset used in steps 4 and 5.

Owner: Fariha.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_rv_dataset(name: str) -> pd.DataFrame:
    """Return a DataFrame with columns: time, mnvel, errvel, tel.

    TODO(Fariha):
      1. Pick ONE published RV dataset and commit to it. A system with a
         moderately eccentric orbit is the informative choice - a circular
         orbit makes the Kepler solver almost irrelevant and the whole
         propagation study would measure nothing.
      2. RadVel ships example datasets (radvel.utils / the example_data
         folder) - using one of those is the lowest-friction option and
         makes our numbers reproducible by anyone with radvel installed.
      3. Put raw files in data/raw/ (git-ignored) and document in
         data/README.md where to download them from. Do NOT commit the data.
      4. Record the source, the instrument(s) and the number of points in
         the docstring - the report has to cite it.
    """
    raise NotImplementedError("load_rv_dataset: see TODO above")


def dataset_path(name: str) -> Path:
    """Path to a raw data file under data/raw/."""
    return Path(__file__).resolve().parents[3] / "data" / "raw" / name
