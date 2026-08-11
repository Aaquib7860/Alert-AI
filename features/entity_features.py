"""Phase 3 -- Entity feature / representation builders.

Master plan section 5 (Entity Intelligence Design) and section 8 (Feature
Engineering Rules). Covers Customer Name Alerts (CustomerViolation) and
Transaction Name Alerts (TransactionNameViolation) -- the two sheets that
share the entity/watchlist-relationship problem.

This module builds *representations*, not decisions: no hand-written
name/DOB/nationality match weight exists anywhere here (master plan
non-negotiable rule). Every representation feeds an anomaly/novelty model
in Phase 5 -- it never itself decides a match.

Only pre-decision fields are touched. Every function here operates on the
Phase 2 normalized output; nothing in `pipelines.validation.temporal.
POST_SCORING_COLUMNS` for the sheet is ever read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from pipelines.validation.temporal import POST_SCORING_COLUMNS, SCORING_TIME_COLUMN

FEATURE_VERSION = "entity-v1"

# Name columns to build a character-n-gram TF-IDF representation for.
# master plan section 5: "may begin with character n-gram TF-IDF as a
# strong, explainable, local, non-LLM baseline."
NAME_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": ["Alerted Party Name (Normalized)", "Hit Details (Name) (Normalized)"],
    "TransactionNameViolation": ["Alerted Party Name (Normalized)", "Hit Details (Name) (Normalized)"],
}

# Context categorical columns -- all low-cardinality (2-41 distinct values,
# checked directly against Alerts_Samples.xlsx), safe for one-hot encoding.
CONTEXT_CATEGORICAL_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": [
        "Customer Type", "Alerted Party", "Alert Type",
        "Sanctions Screening List Name", "Branch Name",
    ],
    "TransactionNameViolation": [
        "Customer Type", "Alerted Party", "Alert Type",
        "Sanctions Screening List Name", "Branch Name",
    ],
}

# Normalized nationality columns (Phase 2 output), used as their own
# categorical family rather than mixed into CONTEXT_CATEGORICAL_COLUMNS --
# these carry a MultiValue/UnresolvedToken flag pair that context columns
# don't have.
NATIONALITY_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": ["Alerted Party Nationality", "Hit Details (Nationality)"],
    "TransactionNameViolation": ["Alerted Party Nationality", "Hit Details (Nationality)"],
}

# Plain numeric features already present after Phase 2 (Matched Screening %
# is explicitly sanctioned as a model feature, not a hardcoded threshold --
# master plan section 7 feature-engineering-rules table).
NUMERIC_COLUMNS: dict[str, list[str]] = {
    "CustomerViolation": ["Matched Screening % (Parsed)"],
    "TransactionNameViolation": ["Matched Screening % (Parsed)"],
}

DOB_YEAR_MAX_PLAUSIBLE_AGE = 120


def _assert_feature_columns_are_not_leakage() -> None:
    """Static guard, run at import time: none of the feature-source column
    lists above may ever reference a post-scoring-time (leakage) field for
    that sheet. Catches the mistake at import, not at scoring time.
    """
    for sheet_name in NAME_COLUMNS:
        leakage = set(POST_SCORING_COLUMNS.get(sheet_name, []))
        all_source_cols = (
            NAME_COLUMNS.get(sheet_name, [])
            + CONTEXT_CATEGORICAL_COLUMNS.get(sheet_name, [])
            + NATIONALITY_COLUMNS.get(sheet_name, [])
            + NUMERIC_COLUMNS.get(sheet_name, [])
        )
        for col in all_source_cols:
            base_col = col.split(" (")[0]
            if base_col in leakage:
                raise AssertionError(
                    f"Entity feature column '{col}' derives from leakage "
                    f"field '{base_col}' for sheet '{sheet_name}'"
                )


_assert_feature_columns_are_not_leakage()


@dataclass
class EntityFeatureArtifacts:
    """Fitted transformers + the assembled feature manifest. Persist this
    (not just the matrix) so scoring-time inference uses the exact same
    vocabulary/categories as training -- an unseen category or unseen
    n-gram at inference must be handled gracefully (handle_unknown), never
    cause a silent shape mismatch.
    """
    sheet_name: str
    feature_version: str
    name_vectorizers: dict[str, TfidfVectorizer]
    categorical_encoders: dict[str, OneHotEncoder]
    nationality_encoders: dict[str, OneHotEncoder]
    feature_names: list[str] = field(default_factory=list)
    fitted_at: str = ""


def _age_years(reference: pd.Series, dob: pd.Series) -> pd.Series:
    """Whole years between dob and reference, NaN where implausible
    (negative or >120 years -- a data-quality problem, not a feature to
    silently clip and pretend is fine) or where either side is missing.
    """
    delta_days = (reference - dob).dt.days
    years = delta_days / 365.25
    years = years.where((years >= 0) & (years <= DOB_YEAR_MAX_PLAUSIBLE_AGE))
    return years


def build_dob_features(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """DOB representation + explicit missingness (master plan section 5:
    "DOB representation + missingness"). Handles the clean `Alerted Party
    DOB (Parsed)` column and the classify-don't-guess `Hit Details (DOB)`
    output from Phase 2 (parsed_date OR year, never both required).
    """
    scoring_col = SCORING_TIME_COLUMN[sheet_name]
    reference = df[f"{scoring_col} (Parsed)"]
    out = pd.DataFrame(index=df.index)

    apdob_col = "Alerted Party DOB (Parsed)"
    if apdob_col in df.columns:
        out["alerted_party_age_years"] = _age_years(reference, df[apdob_col])
        out["alerted_party_dob_missing"] = df[apdob_col].isna().astype(int)

    hit_parsed_col = "Hit Details (DOB) (Parsed)"
    hit_year_col = "Hit Details (DOB) (Year)"
    if hit_parsed_col in df.columns:
        hit_age_from_date = _age_years(reference, df[hit_parsed_col])
        # fall back to year-only resolution where a full date wasn't available
        hit_age_from_year = reference.dt.year - df[hit_year_col].astype("Float64")
        hit_age_from_year = hit_age_from_year.where(
            (hit_age_from_year >= 0) & (hit_age_from_year <= DOB_YEAR_MAX_PLAUSIBLE_AGE)
        )
        out["hit_age_years"] = hit_age_from_date.fillna(hit_age_from_year.astype("float64"))
        out["hit_dob_resolved"] = out["hit_age_years"].notna().astype(int)
        out["hit_dob_multi_value"] = df["Hit Details (DOB) (MultiValue)"].astype(int)
        out["hit_dob_unresolved"] = df["Hit Details (DOB) (Unresolved)"].astype(int)

    return out


def build_historical_customer_context(
    df: pd.DataFrame, sheet_name: str, customer_col: str = "customer_id"
) -> pd.DataFrame:
    """Prior-alert count for this customer, strictly before this alert's
    scoring time (master plan: history features must use "only prior-to-
    alert information in temporal validation"). NaN (not 0) where the
    timestamp or customer id is missing -- we cannot claim "no history" for
    a row we cannot order chronologically at all.
    """
    scoring_col = f"{SCORING_TIME_COLUMN[sheet_name]} (Parsed)"
    valid = df[customer_col].notna() & df[scoring_col].notna()

    counts = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_df = df.loc[valid, [customer_col, scoring_col]].copy()
    # stable sort: ties at the same timestamp keep original row order, so
    # "prior count" is well-defined even when timestamps collide
    valid_df = valid_df.sort_values([customer_col, scoring_col], kind="mergesort")
    valid_df["prior_count"] = valid_df.groupby(customer_col).cumcount()
    counts.loc[valid_df.index] = valid_df["prior_count"]

    return pd.DataFrame({"customer_prior_alert_count": counts}, index=df.index)


def fit_name_vectorizer(names: pd.Series) -> TfidfVectorizer:
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2, lowercase=False,
    )
    vec.fit(names.dropna())
    return vec


def transform_names(vec: TfidfVectorizer, names: pd.Series) -> sp.csr_matrix:
    # TfidfVectorizer.transform requires a str per row -- empty string for
    # missing names yields an all-zero row (correctly "no information"),
    # never a fabricated name.
    return vec.transform(names.fillna("").astype(str))


def fit_categorical_encoder(values: pd.Series) -> OneHotEncoder:
    # handle_unknown="ignore": a category never seen during fit produces an
    # all-zero row at transform time instead of raising -- required for
    # scoring alerts with categories that didn't exist in the training
    # window (master plan section 19: "Tests for unseen categories").
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    enc.fit(values.fillna("MISSING").astype(str).to_frame())
    return enc


def transform_categorical(enc: OneHotEncoder, values: pd.Series) -> sp.csr_matrix:
    return enc.transform(values.fillna("MISSING").astype(str).to_frame())


def fit_entity_feature_artifacts(train_df: pd.DataFrame, sheet_name: str) -> EntityFeatureArtifacts:
    """Fits every transformer on `train_df` only. Callers doing a
    time-forward validation experiment must pass only the training-period
    rows here, then use `transform_entity_features` for both train and
    holdout -- fitting on the full dataset (including holdout) would leak
    vocabulary/category information from the future into the past.
    """
    name_vectorizers = {
        col: fit_name_vectorizer(train_df[col])
        for col in NAME_COLUMNS[sheet_name]
        if col in train_df.columns
    }
    categorical_encoders = {
        col: fit_categorical_encoder(train_df[col])
        for col in CONTEXT_CATEGORICAL_COLUMNS[sheet_name]
        if col in train_df.columns
    }
    nationality_encoders = {
        f"{col} (Normalized)": fit_categorical_encoder(train_df[f"{col} (Normalized)"])
        for col in NATIONALITY_COLUMNS[sheet_name]
        if f"{col} (Normalized)" in train_df.columns
    }

    return EntityFeatureArtifacts(
        sheet_name=sheet_name,
        feature_version=FEATURE_VERSION,
        name_vectorizers=name_vectorizers,
        categorical_encoders=categorical_encoders,
        nationality_encoders=nationality_encoders,
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )


def transform_entity_features(
    df: pd.DataFrame, sheet_name: str, artifacts: EntityFeatureArtifacts
) -> tuple[sp.csr_matrix, list[str]]:
    """Assembles the full entity representation matrix for `df` using
    already-fitted `artifacts`. Returns (sparse_matrix, feature_block_names)
    -- block names describe which columns of the matrix came from which
    feature family, for later interpretability (reason codes, master plan
    API contract).
    """
    blocks: list[sp.csr_matrix] = []
    block_names: list[str] = []

    for col, vec in artifacts.name_vectorizers.items():
        m = transform_names(vec, df[col])
        blocks.append(m)
        block_names.append(f"name_tfidf::{col}")

    for col, enc in artifacts.categorical_encoders.items():
        m = transform_categorical(enc, df[col])
        blocks.append(m)
        block_names.append(f"categorical::{col}")

    for col, enc in artifacts.nationality_encoders.items():
        m = transform_categorical(enc, df[col])
        blocks.append(m)
        block_names.append(f"nationality::{col}")

    numeric_df = df[[c for c in NUMERIC_COLUMNS[sheet_name] if c in df.columns]].fillna(0.0)
    dob_df = build_dob_features(df, sheet_name).fillna(0.0)
    history_df = build_historical_customer_context(df, sheet_name).fillna(-1.0)
    # -1 sentinel for "prior count unknown" -- distinguishable from a real
    # 0 ("known to be this customer's first alert"), never silently 0.

    dense_block = pd.concat([numeric_df, dob_df, history_df], axis=1)
    blocks.append(sp.csr_matrix(dense_block.values))
    block_names.append(f"numeric::{','.join(dense_block.columns)}")

    matrix = sp.hstack(blocks, format="csr")
    return matrix, block_names


def build_entity_features(
    df: pd.DataFrame, sheet_name: str, artifacts: EntityFeatureArtifacts | None = None
) -> tuple[sp.csr_matrix, list[str], EntityFeatureArtifacts]:
    """Convenience one-shot: fits on `df` if no artifacts given (exploratory
    use only -- real experiments must fit on a train split and reuse
    artifacts on the holdout, see fit_entity_feature_artifacts docstring),
    then transforms `df`.
    """
    if artifacts is None:
        artifacts = fit_entity_feature_artifacts(df, sheet_name)
    matrix, block_names = transform_entity_features(df, sheet_name, artifacts)
    artifacts.feature_names = block_names
    return matrix, block_names, artifacts


def save_entity_feature_artifacts(artifacts: EntityFeatureArtifacts, out_dir: Path) -> Path:
    """Persists fitted transformers + a human-readable manifest. The
    manifest alone (no pickle load required) is enough to answer "what
    feature version produced this score, and what did it depend on" for
    audit purposes (master plan: "Every model output must identify model
    version, feature version, schema version").
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{artifacts.sheet_name}_{artifacts.feature_version}.joblib"
    joblib.dump(artifacts, artifact_path)

    manifest = {
        "sheet_name": artifacts.sheet_name,
        "feature_version": artifacts.feature_version,
        "fitted_at": artifacts.fitted_at,
        "name_vectorizer_columns": list(artifacts.name_vectorizers.keys()),
        "name_vocabulary_sizes": {
            col: len(vec.vocabulary_) for col, vec in artifacts.name_vectorizers.items()
        },
        "categorical_columns": list(artifacts.categorical_encoders.keys()),
        "categorical_category_counts": {
            col: sum(len(cats) for cats in enc.categories_)
            for col, enc in artifacts.categorical_encoders.items()
        },
        "nationality_columns": list(artifacts.nationality_encoders.keys()),
        "feature_block_order": artifacts.feature_names,
        "artifact_file": artifact_path.name,
    }
    manifest_path = out_dir / f"{artifacts.sheet_name}_{artifacts.feature_version}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return artifact_path


def load_entity_feature_artifacts(path: Path) -> EntityFeatureArtifacts:
    return joblib.load(path)
