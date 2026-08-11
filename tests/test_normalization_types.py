import numpy as np
import pandas as pd

from pipelines.normalization.types import (
    normalize_categorical,
    normalize_date,
    normalize_id,
    normalize_numeric,
    normalize_text,
    stringify_for_storage,
)


def test_normalize_date_parses_valid_and_coerces_invalid():
    s = pd.Series(["2024-01-15", "not a date", None])
    out = normalize_date(s)
    assert pd.notna(out.iloc[0])
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])


def test_normalize_numeric_coerces_non_numeric_to_nan_not_zero():
    s = pd.Series(["100", "abc", None, "0"])
    out = normalize_numeric(s)
    assert out.iloc[0] == 100.0
    assert np.isnan(out.iloc[1])  # not silently coerced to 0
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == 0.0  # a real zero is preserved as zero, not dropped


def test_normalize_categorical_trims_collapses_whitespace_uppercases():
    s = pd.Series(["  released ", "UPS", "Follow   up"])
    out = normalize_categorical(s)
    assert out.iloc[0] == "RELEASED"
    assert out.iloc[1] == "UPS"
    assert out.iloc[2] == "FOLLOW UP"


def test_normalize_text_collapses_whitespace_preserves_case():
    s = pd.Series(["  Hello   World  ", None])
    out = normalize_text(s)
    assert out.iloc[0] == "Hello World"
    assert pd.isna(out.iloc[1])


def test_normalize_id_strips_trailing_float_artifact():
    s = pd.Series([123.0, "456", None, "789.0"])
    out = normalize_id(s)
    assert out.iloc[0] == "123"
    assert out.iloc[1] == "456"
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == "789"


def test_stringify_for_storage_preserves_null_converts_everything_else():
    s = pd.Series([1, "text", None], dtype=object)
    out = stringify_for_storage(s)
    assert out.iloc[0] == "1"
    assert out.iloc[1] == "text"
    assert pd.isna(out.iloc[2])
