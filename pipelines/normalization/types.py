"""Strict type normalization for dates, numeric fields, categorical values,
IDs and text.

Master plan section 7: "Create strict type normalization for dates, numeric
fields, categorical values, IDs and text." Every function here is a pure,
deterministic, documented transform -- no silent repair of ambiguous values.
Ambiguous/unparseable values become NaT/NaN (explicit missing), never a
guessed value.
"""
from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_date(series: pd.Series) -> pd.Series:
    """Coerce to pandas datetime64[ns]. Already-parsed datetime columns pass
    through; unparseable values become NaT rather than being guessed.
    """
    return pd.to_datetime(series, errors="coerce")


def normalize_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float64. Non-numeric values become NaN rather than 0 or a
    guessed value -- a silently-inserted 0 would be indistinguishable from a
    real zero amount, which is a leakage/correctness risk in itself.
    """
    return pd.to_numeric(series, errors="coerce")


def normalize_categorical(series: pd.Series) -> pd.Series:
    """Trim, collapse internal whitespace, uppercase. Preserves distinct
    categories -- does not merge similar-looking categories (that would be
    a judgment call belonging to a documented business decision, not a
    blanket normalization rule).
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.str.upper()


def normalize_text(series: pd.Series) -> pd.Series:
    """Unicode-normalize (NFKC), trim, collapse whitespace. Case is left
    alone here -- callers needing case-insensitive comparison should upper
    the result themselves; this function is for general free text (names,
    descriptions), not categorical labels.
    """
    def _norm(v):
        if pd.isna(v):
            return v
        v = unicodedata.normalize("NFKC", str(v))
        v = " ".join(v.split())
        return v

    return series.map(_norm)


def stringify_for_storage(series: pd.Series) -> pd.Series:
    """Converts every non-null value to its str() representation.

    Needed for columns pandas loaded as `object` dtype holding a genuine
    mix of native Python types (e.g. datetime.datetime, int, str all in one
    column -- see pipelines/normalization/dob_text.py for why that happens
    here). Parquet/Arrow requires one consistent type per column; this
    keeps the exact display value for audit while making the column
    storable. It is a storage-format cast, not an interpretation -- no
    value is parsed, guessed, or dropped.
    """
    return series.map(lambda v: v if pd.isna(v) else str(v))


def normalize_id(series: pd.Series) -> pd.Series:
    """IDs are stored as-is-string, trimmed, with any trailing '.0' from a
    float-inferred numeric ID column removed (Excel/pandas often infers
    an integer ID column as float64 when any row is blank).
    """
    def _norm(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip()
        if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
            s = s[:-2]
        return s

    return series.map(_norm)
