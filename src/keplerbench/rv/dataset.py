"""Loading the real radial-velocity dataset used in steps 4 and 5.

Owner: Fariha.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


#: name -> loading spec: the raw file, how pandas should read it, a
#: raw-column -> canonical-column rename map, and (only for single-
#: instrument files that don't record one) a fixed telescope label.
_KNOWN_DATASETS: dict[str, dict] = {
    "k2-24": {
        "filename": "k2-24.csv",
        "read_kwargs": {"index_col": 0},
        "rename": {"t": "time", "vel": "mnvel"},
        "tel": "hires",
    },
    "hd164922": {
        "filename": "hd164922.txt",
        "read_kwargs": {"sep": r"\s+"},
        "rename": {},
        "tel": None,  # already has a real, multi-instrument tel column
    },
}


def load_rv_dataset(name: str) -> pd.DataFrame:
    """Return a DataFrame with columns: time, mnvel, errvel, tel.

    Two datasets are available:

    - ``"k2-24"`` (EPIC 203771098): 32 HIRES points, single instrument, a
      two-planet sub-Saturn system.
    - ``"hd164922"``: 401 points across 3 HIRES eras/setups (multi-
      instrument - see ``rv.radvel_bridge.build_posterior`` for how that is
      handled), a multi-planet system.

    See ``data/README.md`` for provenance and citation. Both are real,
    moderately eccentric, multi-planet systems, so the Kepler solver's
    accuracy is not irrelevant to a fit the way it would be for a circular
    orbit.
    """
    key = name
    for ext in (".csv", ".txt"):
        if key.endswith(ext):
            key = key[: -len(ext)]
    if key not in _KNOWN_DATASETS:
        raise KeyError(f"unknown RV dataset {name!r}; known: {sorted(_KNOWN_DATASETS)}")
    spec = _KNOWN_DATASETS[key]

    path = dataset_path(spec["filename"])
    df = pd.read_csv(path, **spec["read_kwargs"])
    df = df.rename(columns=spec["rename"])
    if "tel" not in df.columns:
        df["tel"] = spec["tel"]
    return df[["time", "mnvel", "errvel", "tel"]].reset_index(drop=True)


def dataset_path(name: str) -> Path:
    """Path to a raw data file under data/raw/."""
    return Path(__file__).resolve().parents[3] / "data" / "raw" / name
