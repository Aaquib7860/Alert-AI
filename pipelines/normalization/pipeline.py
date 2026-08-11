"""Phase 2 pipeline orchestration: ingestion -> validation -> normalization
-> versioned persistence -> automatic data-quality report.

Master plan section 7 (Data Pipeline Implementation). Every step here is
additive and traceable: raw columns are never overwritten, only
`(Parsed)`/`(Normalized)` companion columns are added, so any downstream
consumer -- or a human auditor -- can always see exactly what the source
system sent versus what the pipeline derived from it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipelines.ingestion.identifiers import add_stable_ids, find_near_duplicate_candidates
from pipelines.ingestion.load_alerts import get_raw_data_path, load_raw_alerts
from pipelines.normalization import dob_text, geo, names, types
from pipelines.normalization.missingness import add_missingness_indicators
from pipelines.validation.schema_registry import SCHEMA_REGISTRY, SCHEMA_VERSION, validate_schema

DATASET_SEMANTIC_VERSION = "v1"

NAME_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": ["Alerted Party Name", "Hit Details (Name)"],
    "TransactionNameViolation": ["Alerted Party Name", "Hit Details (Name)"],
    "Rule": ["Customer Name", "Beneficiary Name"],
}

COUNTRY_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": ["Alerted Party Nationality", "Hit Details (Nationality)"],
    "TransactionNameViolation": ["Alerted Party Nationality", "Hit Details (Nationality)"],
    "Rule": ["Customer Nationality"],
}

# Key columns used for near-duplicate candidate detection, per sheet --
# normalized-name + parsed-DOB + normalized-nationality where all three
# exist for that sheet's primary alerted-party identity.
# Columns whose real content is NOT a clean date despite the schema calling
# them "date" -- confirmed by direct inspection (see dob_text.py docstring).
# Routed through the classify-don't-guess handler instead of pd.to_datetime.
HETEROGENEOUS_DATE_COLUMNS: dict[str, set[str]] = {
    "CustomerViolation": {"Hit Details (DOB)"},
    "TransactionNameViolation": {"Hit Details (DOB)"},
    "Rule": set(),
}

# IMPORTANT: identity fields alone (name+DOB+nationality, or customer name)
# are NOT enough -- a real repeat customer legitimately produces many alerts
# over time that would share exactly those fields, and flagging every one
# of them as a "near duplicate" would just be re-discovering "this is the
# same person again", not evidence of accidental duplication. An earlier
# version of this key did exactly that and flagged 1,500+/2,397 rows in
# CustomerViolation -- clearly wrong. The key must also pin down the same
# *event* (same alert timestamp / same underlying transaction), so a "near
# duplicate candidate" means "almost certainly the same alert, recorded
# with a trivial textual difference" -- not "same person alerted twice".
NEAR_DUPLICATE_KEYS: dict[str, list[str]] = {
    "CustomerViolation": [
        "Alerted Party Name (Normalized)",
        "Hit Details (Name) (Normalized)",
        "Alert Generated Date & Time (Parsed)",
    ],
    "TransactionNameViolation": [
        "Alerted Party Name (Normalized)",
        "Hit Details (Name) (Normalized)",
        "Alert Generated Date & Time (Parsed)",
    ],
    "Rule": [
        "Customer Name (Normalized)",
        "Reference Number (Normalized)",
        "Rule Name",
    ],
}


def stringify_mixed_type_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parquet/Arrow requires one consistent type per column. Any
    object-dtype column whose non-null values span more than one native
    Python type (seen here: raw source columns mixing datetime.datetime,
    int and str -- e.g. TransactionNameViolation's `Alerted Party DOB`
    storing some rows as real Excel dates and others as the text "1/20/2003")
    is cast to its str() display value for storage. This is a storage-safety
    cast, not an interpretation: no value is parsed, guessed, or dropped,
    and every affected column is reported by name so it stays visible
    rather than silently handled.
    """
    out = df.copy()
    affected = []
    for col in out.columns:
        if out[col].dtype != object:
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        distinct_types = non_null.map(type).nunique()
        if distinct_types > 1:
            out[col] = types.stringify_for_storage(out[col])
            affected.append(col)
    return out, affected


def compute_source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    schema = SCHEMA_REGISTRY[sheet_name]
    out = df.copy()

    country_cols = set(COUNTRY_COLUMNS.get(sheet_name, []))
    name_cols = set(NAME_COLUMNS.get(sheet_name, []))
    heterogeneous_date_cols = HETEROGENEOUS_DATE_COLUMNS.get(sheet_name, set())

    for col_spec in schema.columns:
        col = col_spec.name
        if col not in out.columns:
            continue  # required-column presence already checked by validate_schema

        if col_spec.semantic_type == "date" and col in heterogeneous_date_cols:
            classified = dob_text.classify_hit_dob(out[col])
            out[f"{col} (Parsed)"] = classified["parsed_date"]
            out[f"{col} (Year)"] = classified["year"]
            out[f"{col} (MultiValue)"] = classified["is_multi_value"]
            out[f"{col} (Unresolved)"] = classified["is_unresolved"]
            # raw column mixes native python types (datetime/int/str) which
            # cannot be persisted to parquet as-is -- cast to its display
            # string for storage, see types.stringify_for_storage docstring.
            out[col] = types.stringify_for_storage(out[col])
        elif col_spec.semantic_type == "date":
            out[f"{col} (Parsed)"] = types.normalize_date(out[col])
        elif col_spec.semantic_type == "numeric":
            out[f"{col} (Parsed)"] = types.normalize_numeric(out[col])
        elif col_spec.semantic_type == "id":
            out[f"{col} (Normalized)"] = types.normalize_id(out[col])
        elif col_spec.semantic_type == "categorical":
            if col in country_cols:
                geo_result = geo.normalize_country_field(out[col])
                out[f"{col} (Normalized)"] = geo_result["normalized"]
                out[f"{col} (MultiValue)"] = geo_result["is_multi_value"]
                out[f"{col} (UnresolvedToken)"] = geo_result["unresolved_token_present"]
            else:
                out[f"{col} (Normalized)"] = types.normalize_categorical(out[col])
        elif col_spec.semantic_type == "text":
            if col in name_cols:
                out[f"{col} (Normalized)"] = names.normalize_name(out[col])
            else:
                out[f"{col} (Normalized)"] = types.normalize_text(out[col])
        # semantic_type == "leakage": left completely untouched -- normalizing
        # a leakage field would only make it more tempting to wire in later.

    out = add_missingness_indicators(out, sheet_name)
    out = add_stable_ids(out, sheet_name)

    key_cols = [c for c in NEAR_DUPLICATE_KEYS.get(sheet_name, []) if c in out.columns]
    if key_cols:
        out = find_near_duplicate_candidates(out, key_cols)

    return out


def build_quality_report(
    raw_sheets: dict[str, pd.DataFrame],
    normalized_sheets: dict[str, pd.DataFrame],
    validation_results: dict,
    dataset_version: str,
    source_hash: str,
) -> dict:
    report = {
        "phase": "2_data_pipeline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "schema_version": SCHEMA_VERSION,
        "source_file_sha256": source_hash,
        "sheets": {},
    }

    for name, raw_df in raw_sheets.items():
        norm_df = normalized_sheets[name]
        vr = validation_results[name]

        near_dup_col = "near_duplicate_group_id"
        n_near_dup_candidates = (
            int(norm_df[near_dup_col].notna().sum()) if near_dup_col in norm_df else 0
        )

        unresolved_cols = [
            c for c in norm_df.columns
            if c.endswith("(UnresolvedToken)") or c.endswith("(Unresolved)")
        ]
        unresolved_counts = {
            c: int(norm_df[c].sum()) for c in unresolved_cols if norm_df[c].any()
        }

        report["sheets"][name] = {
            "schema_validation": vr.as_dict(),
            "raw_rows": len(raw_df),
            "raw_cols": len(raw_df.columns),
            "normalized_cols": len(norm_df.columns),
            "exact_duplicate_rows": int(raw_df.duplicated(keep="first").sum()),
            "near_duplicate_candidate_rows": n_near_dup_candidates,
            "unresolved_field_counts": unresolved_counts,
            "customer_id_null_count": int(norm_df["customer_id"].isna().sum()),
        }

    report["overall_status"] = (
        "PASS" if all(v["schema_validation"]["passed"] for v in report["sheets"].values())
        else "FAIL"
    )
    return report


def run_phase2_pipeline(
    repo_root: Path,
    persist: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Runs the full Phase 2 pipeline: load -> validate -> normalize ->
    (optionally) persist versioned parquet + manifest -> build quality report.
    Returns (normalized_sheets, quality_report).
    """
    raw_path = get_raw_data_path()
    raw_sheets = load_raw_alerts(raw_path)
    source_hash = compute_source_hash(raw_path)
    dataset_version = f"{DATASET_SEMANTIC_VERSION}-{source_hash[:8]}"

    validation_results = {
        name: validate_schema(df, name) for name, df in raw_sheets.items()
    }
    failed = {n: v for n, v in validation_results.items() if not v.passed}
    if failed:
        raise ValueError(f"Schema validation failed, refusing to normalize: {failed}")

    normalized_sheets = {
        name: normalize_sheet(df, name) for name, df in raw_sheets.items()
    }

    quality_report = build_quality_report(
        raw_sheets, normalized_sheets, validation_results, dataset_version, source_hash
    )

    storage_stringified_columns: dict[str, list[str]] = {}
    if persist:
        out_dir = repo_root / "data" / "normalized" / dataset_version
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in normalized_sheets.items():
            storable_df, affected = stringify_mixed_type_columns(df)
            storage_stringified_columns[name] = affected
            storable_df.to_parquet(out_dir / f"{name}.parquet", index=False)
        quality_report["storage_stringified_columns"] = storage_stringified_columns
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(quality_report, f, indent=2, default=str)

    return normalized_sheets, quality_report
