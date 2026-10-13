import numpy as np
import pandas as pd
import pytest

from ckd_risk.data import clean
from ckd_risk.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def test_clean_renames_and_types(raw_df):
    df = clean(raw_df)
    assert list(df.columns) == FEATURES + [TARGET]
    assert set(df[TARGET].unique()) <= {0, 1}
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])


def test_clean_fixes_raw_quirks(raw_df):
    df = clean(raw_df)
    # stray tabs and '?' are gone
    for col, allowed in CATEGORICAL_FEATURES.items():
        assert set(df[col].dropna().unique()) <= set(allowed)
    # 'ckd\t' labels are kept, not dropped
    assert len(df) == len(raw_df)


def test_clean_accepts_uci_column_names(raw_df):
    uci = raw_df.drop(columns="id").rename(columns={"wc": "wbcc", "rc": "rbcc", "classification": "class"})
    assert list(clean(uci).columns) == FEATURES + [TARGET]


def test_clean_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        clean(raw_df.drop(columns=["sc"]))


def test_unknown_category_becomes_nan(raw_df):
    df = raw_df.copy()
    df.loc[0, "htn"] = "maybe"
    assert np.isnan(clean(df).loc[0, "hypertension"])
