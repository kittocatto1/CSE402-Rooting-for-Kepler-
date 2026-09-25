from __future__ import annotations

from pathlib import Path

import pandas as pd


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
        "tel": None,
    },
    "k2-131": {
        "filename": "k2-131.txt",
        "read_kwargs": {"sep": r"\s+"},
        "rename": {},
        "tel": None,
    },
}


def load_rv_dataset(name: str) -> pd.DataFrame:
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
    return Path(__file__).resolve().parents[3] / "data" / "raw" / name
