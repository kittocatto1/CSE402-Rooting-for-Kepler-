"""Loading the real radial-velocity dataset used in steps 4 and 5.

Owner: Fariha.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


#: name -> (raw column -> canonical column) rename map, plus a fixed
#: telescope label when the raw file does not record one (single-instrument
#: datasets only report one, so there is nothing to disambiguate).
_KNOWN_DATASETS: dict[str, dict] = {
    "k2-24": {
        "rename": {"t": "time", "vel": "mnvel"},
        "tel": "hires",
    },
}


def load_rv_dataset(name: str) -> pd.DataFrame:
    """Return a DataFrame with columns: time, mnvel, errvel, tel.

    Currently backed by ``k2-24`` (EPIC 203771098): 32 HIRES radial-velocity
    points for a two-planet sub-Saturn system. See ``data/README.md`` for
    provenance and citation. This is a moderately eccentric, multi-planet
    system, so the Kepler solver's accuracy is not irrelevant to the fit the
    way it would be for a circular orbit.
    """
    key = name[:-4] if name.endswith(".csv") else name
    if key not in _KNOWN_DATASETS:
        raise KeyError(f"unknown RV dataset {name!r}; known: {sorted(_KNOWN_DATASETS)}")
    spec = _KNOWN_DATASETS[key]

    path = dataset_path(f"{key}.csv")
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns=spec["rename"])
    if "tel" not in df.columns:
        df["tel"] = spec["tel"]
    return df[["time", "mnvel", "errvel", "tel"]].reset_index(drop=True)


def dataset_path(name: str) -> Path:
    """Path to a raw data file under data/raw/."""
    return Path(__file__).resolve().parents[3] / "data" / "raw" / name
