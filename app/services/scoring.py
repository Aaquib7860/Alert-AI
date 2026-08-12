"""Phase 8 -- Core scoring logic.

Master plan section 13 (API Contract). Every function here operates on
pre-decision fields only -- the raw request payload is validated against
`pipelines.validation.schema_registry`, which already excludes every
leakage-typed column (Comment, Maker Comment, Status, etc.) from what a
caller is even allowed to submit as a *feature* -- see
`ALLOWED_REQUEST_FIELDS`.

Score interpretation: the underlying model (One-Class SVM for both current
champions) produces a raw score whose scale is NOT stationary across
retrains (Phase 7 finding). This module never returns that raw score
directly -- it reports a percentile against the champion's training-score
distribution instead: bounded [0, 100], comparable across requests, and
explicitly NOT a probability (master plan: "Never describe an anomaly
score as a probability unless a separate calibration procedure proves that
interpretation" -- no such procedure has been run).
"""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import os

from app.services.model_registry import ALERT_TYPE_TO_SHEET, LoadedModel, load_active_model
from features.entity_features import transform_entity_features
from features.transaction_features import transform_transaction_features
from pipelines.entity.anomaly_models import extract_name_representation_matrix, score_ocsvm
from pipelines.normalization.pipeline import normalize_sheet
from pipelines.validation.schema_registry import SCHEMA_REGISTRY

SCORE_FN_BY_MODEL_KIND = {
    "ocsvm": score_ocsvm,
}

# PoC placeholder -- NOT a validated production threshold (master plan
# section 13/15: routing thresholds must be calibrated and approved, not
# hardcoded from a sample guess). Overridable via environment for the
# shadow-mode/demo period; the response always states this explicitly.
REVIEW_PERCENTILE_THRESHOLD = float(os.environ.get("REVIEW_PERCENTILE_THRESHOLD", "80"))


class AlertValidationError(ValueError):
    def __init__(self, message: str, missing_required_columns: list[str], unexpected_columns: list[str]):
        super().__init__(message)
        self.missing_required_columns = missing_required_columns
        self.unexpected_columns = unexpected_columns


def allowed_request_fields(alert_type: str) -> list[str]:
    """Every non-leakage column for this alert type -- i.e. everything the
    scoring API will accept. Leakage-typed columns (Comment, Status, etc.)
    are excluded here, at the schema level, not filtered out ad hoc later.
    """
    sheet_name = ALERT_TYPE_TO_SHEET[alert_type]
    schema = SCHEMA_REGISTRY[sheet_name]
    return [c.name for c in schema.columns if c.semantic_type != "leakage"]


# Required fields for a *live scoring request* -- deliberately NOT the same
# concept as schema_registry's `required` flag, which answers "must this
# COLUMN exist in the uploaded sheet" (a batch/structural question) not
# "must this VALUE be present for a single alert." Grounded directly in
# Phase 1 per-field missingness on the real workbook: a field is listed
# here only if it was ~0% missing across the historical sample. Fields
# with nonzero missingness (e.g. Alerted Party DOB: 1.8%/78.8% across the
# two name-alert sheets, Hit Details (DOB)/(Nationality), Rule's Customer
# Number/Branch Code/Customer Currency) are deliberately left optional --
# treating them as required would reject a large share of genuine alerts.
LIVE_REQUIRED_FIELDS: dict[str, list[str]] = {
    "CustomerViolation": [
        "UIN", "Customer Type", "Alerted Party Name", "Alerted Party Nationality",
        "Hit Details (Name)", "Matched Screening %", "Sanctions Screening List Name",
        "Alerted Party", "Branch Name", "Alert Generated Date & Time", "Alert Type",
    ],
    "TransactionNameViolation": [
        "UIN", "Customer Type", "Trxn Type", "Trxn date & Time", "Trxn Ref Number",
        "Transaction Status", "Alerted Party Name", "Hit Details (Name)",
        "Matched Screening %", "Sanctions Screening List Name", "Alerted Party",
        "Branch Name", "Alert Generated Date & Time", "Alert Type",
    ],
    "Rule": [
        "Branch Description", "Transaction Type Code", "Reference Number",
        "Transaction Date", "Customer Name", "Customer Nationality",
        "Scan Date", "Rule Type", "Rule Name",
    ],
}


def _required_live_fields(sheet_name: str) -> list[str]:
    return LIVE_REQUIRED_FIELDS.get(sheet_name, [])


def validate_live_request(sheet_name: str, raw_fields: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validates against what the caller actually provided, BEFORE any
    defaulting -- checking presence on a fully-defaulted DataFrame (as
    `pipelines.validation.schema_registry.validate_schema` does for batch
    ingestion) would trivially always pass here, since every column key
    would already exist with a None value. Returns (missing_required,
    unexpected_fields).
    """
    schema = SCHEMA_REGISTRY[sheet_name]
    known_cols = {c.name for c in schema.columns}
    required = set(_required_live_fields(sheet_name))

    provided = {k for k, v in raw_fields.items() if v is not None}
    missing = sorted(required - provided)
    unexpected = sorted(set(raw_fields.keys()) - known_cols)
    return missing, unexpected


def _score_fn(model_kind: str):
    if model_kind not in SCORE_FN_BY_MODEL_KIND:
        raise ValueError(f"No score function wired up for model_kind={model_kind!r}")
    return SCORE_FN_BY_MODEL_KIND[model_kind]


def _percentile(value: float, sorted_distribution: list[float]) -> float:
    """Percentage of the reference training distribution at or below
    `value`. Pure bisect -- O(log n), no scipy dependency needed for this.
    """
    if not sorted_distribution:
        return float("nan")
    idx = bisect.bisect_left(sorted_distribution, value)
    return round(100.0 * idx / len(sorted_distribution), 2)


def _build_single_row_df(sheet_name: str, raw_fields: dict[str, Any]) -> pd.DataFrame:
    schema = SCHEMA_REGISTRY[sheet_name]
    expected_cols = {c.name for c in schema.columns}
    row = {col: raw_fields.get(col) for col in expected_cols}
    return pd.DataFrame([row])


def _reason_codes(sheet_name: str, normalized_row: pd.Series) -> list[str]:
    """Human-readable, feature-level observations -- descriptive labels
    for context, never a second hidden decision engine (master plan
    non-negotiable rule: no hand-weighted heuristic engine). The score and
    recommendation come entirely from the trained model above; these codes
    only explain what stood out in the input, using the same feature
    values the model already saw.
    """
    codes = []
    if sheet_name in ("CustomerViolation", "TransactionNameViolation"):
        if normalized_row.get("Alerted Party DOB_missing"):
            codes.append("ALERTED_PARTY_DOB_MISSING")
        if normalized_row.get("Hit Details (DOB) (Unresolved)"):
            codes.append("HIT_DOB_UNRESOLVED")
        if normalized_row.get("Hit Details (DOB) (MultiValue)"):
            codes.append("HIT_DOB_MULTI_VALUE")
        if normalized_row.get("Hit Details (Nationality)_missing"):
            codes.append("HIT_NATIONALITY_MISSING")
        pct = normalized_row.get("Matched Screening % (Parsed)")
        if pct is not None and not pd.isna(pct) and pct >= 95:
            codes.append("HIGH_SCREENING_MATCH_PERCENTAGE")
    else:  # Rule
        if normalized_row.get("Beneficiary Name_missing"):
            codes.append("BENEFICIARY_NAME_MISSING")
        if normalized_row.get("Beneficiary Relationship_missing"):
            codes.append("BENEFICIARY_RELATIONSHIP_MISSING")
        if normalized_row.get("Currency Name_missing"):
            codes.append("CURRENCY_NAME_MISSING")
    return codes


def _clean_value(v):
    """None for any missing/NaN/NaT scalar, native Python type otherwise
    (numpy scalars aren't JSON-serializable as-is)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass  # v isn't a type pd.isna can evaluate (e.g. a list) -- not expected here, pass through
    return v.item() if hasattr(v, "item") else v


def _evidence_fields(sheet_name: str, normalized_row: pd.Series) -> dict:
    """The specific underlying field values behind the reason codes and
    score -- not just a flag, the actual data -- so a human reviewer can
    look at the same record the model saw and judge it themselves. This is
    supporting evidence for human judgment, not a second decision engine:
    nothing here feeds back into the score or recommendation.
    """
    if sheet_name in ("CustomerViolation", "TransactionNameViolation"):
        return {
            "matched_screening_pct": _clean_value(normalized_row.get("Matched Screening % (Parsed)")),
            "alerted_party_nationality": _clean_value(normalized_row.get("Alerted Party Nationality (Normalized)")),
            "hit_nationality": _clean_value(normalized_row.get("Hit Details (Nationality) (Normalized)")),
            "sanctions_screening_list": _clean_value(normalized_row.get("Sanctions Screening List Name (Normalized)")),
            "alert_type": _clean_value(normalized_row.get("Alert Type (Normalized)")),
        }
    return {
        "rule_name": _clean_value(normalized_row.get("Rule Name (Normalized)")),
        "transaction_type": _clean_value(normalized_row.get("Transaction Type Code (Normalized)")),
        "currency": _clean_value(normalized_row.get("Currency Name (Normalized)")),
        "beneficiary_relationship": _clean_value(normalized_row.get("Beneficiary Relationship (Normalized)")),
        "customer_nationality": _clean_value(normalized_row.get("Customer Nationality (Normalized)")),
    }


# Plain-English translation for each internal reason code -- for UI
# display only, the internal code is still returned in `reason_codes`.
REASON_CODE_PLAIN_TEXT: dict[str, str] = {
    "ALERTED_PARTY_DOB_MISSING": "Date of birth is missing for the alerted party.",
    "HIT_DOB_UNRESOLVED": "The watchlist hit's date of birth could not be read reliably.",
    "HIT_DOB_MULTI_VALUE": "The watchlist hit lists more than one possible date of birth.",
    "HIT_NATIONALITY_MISSING": "The watchlist hit's nationality is missing.",
    "HIGH_SCREENING_MATCH_PERCENTAGE": "The name-matching percentage from the screening system is very high.",
    "BENEFICIARY_NAME_MISSING": "The beneficiary's name is missing.",
    "BENEFICIARY_RELATIONSHIP_MISSING": "The relationship to the beneficiary is missing.",
    "CURRENCY_NAME_MISSING": "The transaction currency is missing.",
}

# Gray-zone width (percentile points) below the REVIEW threshold that reads
# as "borderline" rather than confidently routine. Same PoC-placeholder
# status as REVIEW_PERCENTILE_THRESHOLD itself -- not calibrated/approved.
BORDERLINE_BAND_WIDTH = 15.0


def _plain_language_summary(global_percentile: float, review_threshold: float) -> dict:
    """Three-tier, everyday-language summary of the score for UI display.

    Deliberately avoids the words "match"/"not a match" anywhere -- master
    plan section 5/12.1: the model never determines whether a name/entity
    is a genuine sanctions match, only how unusual the alert looks against
    historical patterns. "Needs Review" / "Not Confident" / "Looks
    Routine" describe the *recommendation*, not a match verdict. This is
    presentation only -- `recommendation` (REVIEW/LOWER_TOUCH_CANDIDATE)
    remains the authoritative field; this never changes what gets routed.
    """
    if global_percentile >= review_threshold:
        return {
            "label": "Needs Review",
            "detail": "This alert looks unusual compared to historical patterns.",
        }
    if global_percentile >= review_threshold - BORDERLINE_BAND_WIDTH:
        return {
            "label": "Not Confident",
            "detail": "This alert is borderline -- not clearly usual or unusual.",
        }
    return {
        "label": "Looks Routine",
        "detail": "This alert looks similar to historical patterns.",
    }


def score_alert(alert_type: str, alert_id: str, raw_fields: dict[str, Any]) -> dict:
    if alert_type not in ALERT_TYPE_TO_SHEET:
        raise ValueError(f"Unknown alert_type {alert_type!r}. Must be one of {list(ALERT_TYPE_TO_SHEET)}")

    sheet_name = ALERT_TYPE_TO_SHEET[alert_type]

    missing, unexpected = validate_live_request(sheet_name, raw_fields)
    if missing:
        raise AlertValidationError(
            f"Alert missing required field(s) for {sheet_name}: {missing}",
            missing_required_columns=missing,
            unexpected_columns=unexpected,
        )

    row_df = _build_single_row_df(sheet_name, raw_fields)
    normalized = normalize_sheet(row_df, sheet_name)
    normalized_row = normalized.iloc[0]

    loaded: LoadedModel = load_active_model(alert_type)

    if alert_type in ("customer_name", "transaction_name"):
        normalized["alert_source_sheet"] = sheet_name
        matrix, _ = transform_entity_features(normalized, "CombinedEntity", loaded.feature_artifacts)
        representation_matrix = extract_name_representation_matrix(matrix, loaded.feature_artifacts)
    else:
        matrix, _ = transform_transaction_features(normalized, loaded.feature_artifacts)
        representation_matrix = matrix

    X = loaded.svd.transform(representation_matrix) if loaded.svd is not None else representation_matrix

    score_fn = _score_fn(loaded.model_kind)
    raw_score = float(score_fn(loaded.model, X)[0])

    global_percentile = _percentile(raw_score, loaded.train_score_distribution)

    customer_id = normalized_row.get("customer_id")
    customer_baseline = loaded.customer_baselines.get(customer_id) if customer_id else None
    customer_novelty = None
    if customer_baseline is not None and customer_baseline["std"] > 0:
        customer_novelty = round((raw_score - customer_baseline["mean"]) / customer_baseline["std"], 3)

    recommendation = "REVIEW" if global_percentile >= REVIEW_PERCENTILE_THRESHOLD else "LOWER_TOUCH_CANDIDATE"
    plain_language = _plain_language_summary(global_percentile, REVIEW_PERCENTILE_THRESHOLD)

    reason_codes = _reason_codes(sheet_name, normalized_row)
    reason_codes_plain = [REASON_CODE_PLAIN_TEXT.get(c, c) for c in reason_codes]
    evidence = _evidence_fields(sheet_name, normalized_row)

    historical_context = {
        "customer_prior_alert_count": (
            int(customer_baseline["n"]) if customer_baseline is not None else 0
        ),
        "customer_baseline_known": customer_baseline is not None,
    }

    return {
        "alert_id": alert_id,
        "alert_type": alert_type,
        "model_version": loaded.model_version,
        "feature_version": loaded.feature_version,
        "schema_version": loaded.schema_version,
        "novelty": {
            "global": global_percentile,
            "customer": customer_novelty,
        },
        "novelty_scale_note": (
            "global = percentile rank (0-100) against the champion's training score "
            "distribution; NOT a probability. customer = z-score vs. this customer's own "
            "training-period scores where a baseline exists, else null."
        ),
        "recommendation": recommendation,
        "recommendation_threshold_note": (
            f"PoC placeholder threshold (global percentile >= {REVIEW_PERCENTILE_THRESHOLD}) -- "
            "not yet calibrated or approved for production routing (master plan section 13/15)."
        ),
        "plain_language_label": plain_language["label"],
        "plain_language_detail": plain_language["detail"],
        "reason_codes": reason_codes,
        "reason_codes_plain": reason_codes_plain,
        "evidence": evidence,
        "historical_context": historical_context,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
