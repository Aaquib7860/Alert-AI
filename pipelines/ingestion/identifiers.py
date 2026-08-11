"""Stable identifiers for grouped/temporal validation, and near-duplicate
candidate detection.

Master plan section 7: "Create stable customer/UIN, transaction-reference
and entity identifiers" and "Detect exact duplicates and near-duplicate
candidates separately."

Important scope note: the three sheets come from different source systems
(name-screening vs. rule engine). `UIN` in CustomerViolation/
TransactionNameViolation and `Customer Number` in Rule are NOT verified to
share the same identifier space -- the client has not confirmed this
crosswalk. IDs here are namespaced per sheet (e.g. "customerviolation:1234")
specifically so nothing downstream can accidentally join across sheets on a
bare numeric match that hasn't been proven meaningful.
"""
from __future__ import annotations

import hashlib

import pandas as pd

# (sheet_name -> (customer key column, transaction key column or None))
ID_KEY_COLUMNS: dict[str, tuple[str, str | None]] = {
    "CustomerViolation": ("UIN", None),
    "TransactionNameViolation": ("UIN", "Trxn Ref Number"),
    "Rule": ("Customer Number", "Reference Number"),
}


def _namespaced_id(sheet_name: str, kind: str, value) -> object:
    if pd.isna(value):
        return pd.NA
    v = str(value)
    if v.endswith(".0") and v[:-2].lstrip("-").isdigit():
        v = v[:-2]
    return f"{sheet_name.lower()}:{kind}:{v}"


def add_stable_ids(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Adds `customer_id` (always) and `transaction_id` (where the sheet has
    a transaction reference) as namespaced string columns, plus a
    content-hash `record_id` that is stable across reloads regardless of row
    order -- useful for tracing a scored result back to its source row
    without depending on positional index.
    """
    if sheet_name not in ID_KEY_COLUMNS:
        raise KeyError(f"No ID key columns registered for sheet '{sheet_name}'")

    customer_col, txn_col = ID_KEY_COLUMNS[sheet_name]
    out = df.copy()

    out["customer_id"] = out[customer_col].map(
        lambda v: _namespaced_id(sheet_name, "customer", v)
    )
    if txn_col is not None:
        out["transaction_id"] = out[txn_col].map(
            lambda v: _namespaced_id(sheet_name, "txn", v)
        )

    def _record_hash(row: pd.Series) -> str:
        payload = "|".join(str(v) for v in row.values)
        return hashlib.sha1(f"{sheet_name}|{payload}".encode("utf-8")).hexdigest()[:16]

    # hash over the ORIGINAL columns only, so record_id is stable even if
    # later pipeline steps add more derived columns to the right
    original_cols = list(df.columns)
    out["record_id"] = df[original_cols].apply(_record_hash, axis=1)

    return out


def find_near_duplicate_candidates(
    df: pd.DataFrame, key_columns: list[str], exclude_exact_duplicates: bool = True
) -> pd.DataFrame:
    """Flags rows whose values are identical across `key_columns` (e.g.
    normalized name + DOB + nationality) but the full row is NOT an exact
    duplicate -- i.e. some other field differs. This is exact-match grouping
    on already-normalized columns, not a fuzzy-distance/similarity engine,
    to stay inside the "no RapidFuzz/Levenshtein/phonetic matching" rule:
    it only flags candidates for review, it never decides a match.

    Returns a DataFrame with an added `near_duplicate_group_id` column
    (NA where the row has no near-duplicate candidates).
    """
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise KeyError(f"key_columns not found in DataFrame: {missing}")

    out = df.copy()
    valid_key_mask = out[key_columns].notna().all(axis=1)

    exact_dup_mask = pd.Series(False, index=out.index)
    if exclude_exact_duplicates:
        exact_dup_mask = out.duplicated(keep=False)

    candidates = out[valid_key_mask & ~exact_dup_mask]
    group_sizes = candidates.groupby(key_columns, dropna=False).size()
    group_ids = {
        key: idx for idx, key in enumerate(group_sizes[group_sizes > 1].index)
    }

    def _assign_group(row) -> object:
        if not valid_key_mask.loc[row.name] or exact_dup_mask.loc[row.name]:
            return pd.NA
        key = tuple(row[c] for c in key_columns) if len(key_columns) > 1 else row[key_columns[0]]
        return group_ids.get(key, pd.NA)

    out["near_duplicate_group_id"] = out.apply(_assign_group, axis=1)
    return out
