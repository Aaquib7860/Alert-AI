"""Name normalization for representation generation.

Master plan section 7: "Normalize names for representation generation,
preserving the raw value for audit." The normalized form is for building
character/token representations later (Phase 3) -- it is never itself a
matching decision (master plan non-negotiable rule: no hand-written
similarity thresholds as the decision engine).
"""
from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_name(series: pd.Series) -> pd.Series:
    """Unicode NFKC normalize, trim, collapse internal whitespace, uppercase.
    Deterministic and reversible-in-spirit -- the raw column must always be
    kept alongside this one; this function does not touch the source data.
    """
    def _norm(v):
        if pd.isna(v):
            return v
        v = unicodedata.normalize("NFKC", str(v))
        v = " ".join(v.split())
        return v.upper()

    return series.map(_norm)


def add_normalized_name_column(df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    """Returns a copy of df with an added f"{raw_col} (Normalized)" column.
    The raw column is left untouched -- audit must always be able to trace
    a normalized value back to exactly what the source system sent.
    """
    out = df.copy()
    out[f"{raw_col} (Normalized)"] = normalize_name(out[raw_col])
    return out
