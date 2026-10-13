"""Loading and cleaning the raw CKD data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, COLUMN_MAP, FEATURES, NUMERIC_FEATURES, TARGET


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with readable column names, typed values and a 0/1 target.

    The raw files contain stray tabs and spaces (``"\\tyes"``, ``"ckd\\t"``),
    ``"?"`` for missing values, and numbers stored as text. Values outside the
    known categories become NaN so the model's imputer handles them.
    """
    df = raw.rename(columns={c: COLUMN_MAP.get(str(c).strip().lower(), c) for c in raw.columns})
    df = df.drop(columns=[c for c in ("id",) if c in df.columns])

    missing = [c for c in FEATURES + ["class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")

    df = df[FEATURES + ["class"]].copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower().replace({"?": np.nan, "nan": np.nan, "": np.nan})

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, allowed in CATEGORICAL_FEATURES.items():
        df[col] = df[col].where(df[col].isin(allowed))

    label = df.pop("class").astype(str).str.strip().str.lower()
    df[TARGET] = label.map({"ckd": 1, "notckd": 0, "not ckd": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def load(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a CSV of raw data and split it into features and target."""
    df = clean(pd.read_csv(path))
    return df[FEATURES], df[TARGET]
