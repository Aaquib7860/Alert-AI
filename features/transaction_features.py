"""Phase 4 -- Transaction / Rule feature representation builders.

Master plan section 6 (Transaction Intelligence Design). Covers the `Rule`
sheet (Transaction Rule Alerts) -- the transaction-monitoring / behavioural
pipeline, separate from the entity/watchlist pipeline in
`features/entity_features.py`. Rule alerts "primarily concern transaction
and customer behavioural abnormality" (feasibility report section 2), not
name matching, so there is no character-n-gram name representation here
except for `Beneficiary Name`, which the master plan explicitly lists as a
"use when present" context feature -- not a match key.

Master plan: "Do not train 18 independent rule models in the first PoC. Use
a shared transaction representation with rule context." Rule Name/Type is
encoded as one more context feature among the 18 values, not 18 separate
pipelines.

**Transaction amount is deliberately excluded.** The Rule sheet has no
clean structured amount field (confirmed in Phase 1). Amounts appear only
inside `Comment` -- which is itself a leakage field (post-review text,
master plan non-negotiable rule 12) -- so extracting an amount from it
would smuggle leakage into a feature under a different name, not just
produce a low-quality feature. Master plan section 1 rule 16 is explicit:
"Do not make production transaction amount depend on parsing comments."
This module has no code path that reads `Comment` at all. Amount becomes
available only once the client supplies a structured source field
(feasibility report section 14).
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

from features.entity_features import (
    build_historical_customer_context,
    fit_categorical_encoder,
    fit_name_vectorizer,
    transform_categorical,
    transform_names,
)
from pipelines.validation.temporal import POST_SCORING_COLUMNS, SCORING_TIME_COLUMN

FEATURE_VERSION = "transaction-v1"
SHEET_NAME = "Rule"

# Context categorical columns -- all grounded in observed cardinality
# (6-50 distinct values, checked directly against Alerts_Samples.xlsx).
# `Customer Currency` is deliberately excluded: it is a constant
# ("UAE DIRHAM" in every row of the current sample) and would contribute
# zero signal while adding a column -- documented, not silently dropped.
# `Purpose Code` is excluded as redundant with `Purpose` (same cardinality,
# same rows populated) -- encoding both would just duplicate one signal.
CONTEXT_CATEGORICAL_COLUMNS = [
    "Transaction Type Code",
    "Branch Description",
    "Currency Name",
    "Beneficiary Relationship",
    "Purpose",
    "Rule Name",
]

# Rule's Customer Nationality is already a clean full-name field in this
# workbook (no "XX-NAME" prefix, no multi-value lists -- unlike the name-
# alert sheets' nationality columns) so the Phase 2 generic categorical
# normalization is sufficient; no need for the geo multi-value parser here.
NATIONALITY_COLUMN = "Customer Nationality (Normalized)"

BENEFICIARY_NAME_COLUMN = "Beneficiary Name (Normalized)"


@dataclass
class TransactionFeatureArtifacts:
    feature_version: str
    categorical_encoders: dict[str, OneHotEncoder]
    nationality_encoder: OneHotEncoder | None
    beneficiary_name_vectorizer: TfidfVectorizer | None
    feature_names: list[str] = field(default_factory=list)
    fitted_at: str = ""


def _assert_feature_columns_are_not_leakage() -> None:
    leakage = set(POST_SCORING_COLUMNS.get(SHEET_NAME, []))
    all_cols = CONTEXT_CATEGORICAL_COLUMNS + [NATIONALITY_COLUMN, BENEFICIARY_NAME_COLUMN]
    for col in all_cols:
        base_col = col.split(" (")[0]
        if base_col in leakage:
            raise AssertionError(
                f"Transaction feature column '{col}' derives from leakage "
                f"field '{base_col}'"
            )


_assert_feature_columns_are_not_leakage()


def build_prior_rule_diversity(
    df: pd.DataFrame, customer_col: str = "customer_id", rule_col: str = "Rule Name"
) -> pd.DataFrame:
    """For each row, the count of *distinct* rule names this customer
    triggered strictly before this alert's scoring time (Scan Date). A
    customer repeatedly triggering the same one rule looks different from a
    customer triggering many different rules -- this distinguishes them.
    Uses the same strictly-prior, stable-sort discipline as
    `entity_features.build_historical_customer_context` (see that
    docstring for the same-timestamp tie-breaking caveat).
    """
    scoring_col = f"{SCORING_TIME_COLUMN[SHEET_NAME]} (Parsed)"
    valid = df[customer_col].notna() & df[scoring_col].notna() & df[rule_col].notna()

    diversity = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_df = df.loc[valid, [customer_col, scoring_col, rule_col]].copy()
    valid_df = valid_df.sort_values([customer_col, scoring_col], kind="mergesort")

    # explicit per-group loop -- avoids groupby().apply()'s ambiguous
    # Series-vs-DataFrame return type across pandas versions when a group
    # has a single row or spans the whole frame
    counts_by_index: dict = {}
    for _, group in valid_df.groupby(customer_col, sort=False):
        seen: set = set()
        for idx, rule_name in group[rule_col].items():
            counts_by_index[idx] = len(seen)
            seen.add(rule_name)

    diversity.loc[list(counts_by_index.keys())] = list(counts_by_index.values())

    return pd.DataFrame({"customer_prior_distinct_rule_count": diversity}, index=df.index)


def fit_transaction_feature_artifacts(train_df: pd.DataFrame) -> TransactionFeatureArtifacts:
    """Fits on `train_df` only -- see entity_features.fit_entity_feature_artifacts
    docstring for why (time-forward experiments must not fit on holdout rows).
    """
    categorical_encoders = {
        col: fit_categorical_encoder(train_df[col])
        for col in CONTEXT_CATEGORICAL_COLUMNS
        if col in train_df.columns
    }
    nationality_encoder = (
        fit_categorical_encoder(train_df[NATIONALITY_COLUMN])
        if NATIONALITY_COLUMN in train_df.columns else None
    )
    beneficiary_vectorizer = (
        fit_name_vectorizer(train_df[BENEFICIARY_NAME_COLUMN])
        if BENEFICIARY_NAME_COLUMN in train_df.columns else None
    )

    return TransactionFeatureArtifacts(
        feature_version=FEATURE_VERSION,
        categorical_encoders=categorical_encoders,
        nationality_encoder=nationality_encoder,
        beneficiary_name_vectorizer=beneficiary_vectorizer,
        fitted_at=datetime.now(timezone.utc).isoformat(),
    )


def transform_transaction_features(
    df: pd.DataFrame, artifacts: TransactionFeatureArtifacts
) -> tuple[sp.csr_matrix, list[str]]:
    blocks: list[sp.csr_matrix] = []
    block_names: list[str] = []

    for col, enc in artifacts.categorical_encoders.items():
        blocks.append(transform_categorical(enc, df[col]))
        block_names.append(f"categorical::{col}")

    if artifacts.nationality_encoder is not None:
        blocks.append(transform_categorical(artifacts.nationality_encoder, df[NATIONALITY_COLUMN]))
        block_names.append(f"nationality::{NATIONALITY_COLUMN}")

    if artifacts.beneficiary_name_vectorizer is not None:
        blocks.append(transform_names(artifacts.beneficiary_name_vectorizer, df[BENEFICIARY_NAME_COLUMN]))
        block_names.append(f"name_tfidf::{BENEFICIARY_NAME_COLUMN}")

    numeric_parts = []
    numeric_names = []

    for missing_col in [
        "Beneficiary Name_missing", "Beneficiary Id Number_missing",
        "Beneficiary Relationship_missing", "Currency Name_missing", "Purpose_missing",
    ]:
        if missing_col in df.columns:
            numeric_parts.append(df[missing_col].astype(int))
            numeric_names.append(missing_col)

    history = build_historical_customer_context(df, SHEET_NAME)
    numeric_parts.append(history["customer_prior_alert_count"].fillna(-1.0))
    numeric_names.append("customer_prior_alert_count")

    diversity = build_prior_rule_diversity(df)
    numeric_parts.append(diversity["customer_prior_distinct_rule_count"].fillna(-1.0))
    numeric_names.append("customer_prior_distinct_rule_count")

    numeric_df = pd.concat(numeric_parts, axis=1)
    numeric_df.columns = numeric_names
    blocks.append(sp.csr_matrix(numeric_df.values))
    block_names.append(f"numeric::{','.join(numeric_names)}")

    matrix = sp.hstack(blocks, format="csr")
    return matrix, block_names


def build_transaction_features(
    df: pd.DataFrame, artifacts: TransactionFeatureArtifacts | None = None
) -> tuple[sp.csr_matrix, list[str], TransactionFeatureArtifacts]:
    if artifacts is None:
        artifacts = fit_transaction_feature_artifacts(df)
    matrix, block_names = transform_transaction_features(df, artifacts)
    artifacts.feature_names = block_names
    return matrix, block_names, artifacts


def save_transaction_feature_artifacts(artifacts: TransactionFeatureArtifacts, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{SHEET_NAME}_{artifacts.feature_version}.joblib"
    joblib.dump(artifacts, artifact_path)

    manifest = {
        "sheet_name": SHEET_NAME,
        "feature_version": artifacts.feature_version,
        "fitted_at": artifacts.fitted_at,
        "categorical_columns": list(artifacts.categorical_encoders.keys()),
        "categorical_category_counts": {
            col: sum(len(cats) for cats in enc.categories_)
            for col, enc in artifacts.categorical_encoders.items()
        },
        "nationality_column": NATIONALITY_COLUMN if artifacts.nationality_encoder else None,
        "beneficiary_name_vocabulary_size": (
            len(artifacts.beneficiary_name_vectorizer.vocabulary_)
            if artifacts.beneficiary_name_vectorizer else None
        ),
        "excluded_features": {
            "transaction_amount": "no clean structured field in current sample; "
                "amount only appears in the leakage-typed Comment field, which "
                "this module never reads. Requires client-supplied structured field.",
            "customer_currency": "constant in current sample, zero signal.",
            "purpose_code": "redundant with Purpose (same cardinality/coverage).",
        },
        "feature_block_order": artifacts.feature_names,
        "artifact_file": artifact_path.name,
    }
    manifest_path = out_dir / f"{SHEET_NAME}_{artifacts.feature_version}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return artifact_path


def load_transaction_feature_artifacts(path: Path) -> TransactionFeatureArtifacts:
    return joblib.load(path)
