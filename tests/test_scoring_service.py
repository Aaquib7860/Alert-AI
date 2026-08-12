import math

import pandas as pd
import pytest

from app.services.scoring import (
    AlertValidationError,
    LIVE_REQUIRED_FIELDS,
    REASON_CODE_PLAIN_TEXT,
    _percentile,
    _plain_language_summary,
    _reason_codes,
    allowed_request_fields,
    score_alert,
    validate_live_request,
)
from app.services.model_registry import get_models_root
from pipelines.ingestion.load_alerts import load_raw_alerts

REGISTRY_AVAILABLE = (get_models_root() / "entity" / "scoring_registry.json").exists() and (
    get_models_root() / "transaction" / "scoring_registry.json"
).exists()


def test_allowed_request_fields_excludes_leakage():
    fields = allowed_request_fields("customer_name")
    assert "Maker Comment" not in fields
    assert "Alert Status" not in fields
    assert "Alert Closure Date & Time" not in fields
    assert "Alerted Party Name" in fields


def test_allowed_request_fields_unknown_alert_type_raises():
    with pytest.raises(KeyError):
        allowed_request_fields("not_a_real_type")


def test_validate_live_request_flags_missing_required():
    missing, unexpected = validate_live_request("CustomerViolation", {"UIN": 1})
    assert set(LIVE_REQUIRED_FIELDS["CustomerViolation"]) - {"UIN"} <= set(missing)


def test_validate_live_request_none_value_counts_as_missing():
    payload = {f: None for f in LIVE_REQUIRED_FIELDS["CustomerViolation"]}
    missing, _ = validate_live_request("CustomerViolation", payload)
    assert set(missing) == set(LIVE_REQUIRED_FIELDS["CustomerViolation"])


def test_validate_live_request_flags_unexpected_field():
    payload = {"NotARealColumn": "x"}
    _, unexpected = validate_live_request("CustomerViolation", payload)
    assert "NotARealColumn" in unexpected


def test_validate_live_request_never_requires_leakage_fields():
    # sanity: no leakage-typed field should ever appear in the required list
    from pipelines.validation.schema_registry import SCHEMA_REGISTRY

    for sheet_name, required in LIVE_REQUIRED_FIELDS.items():
        leakage = set(SCHEMA_REGISTRY[sheet_name].leakage_columns)
        assert not (set(required) & leakage)


def test_percentile_bounds():
    dist = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(0.0, dist) == 0.0
    assert _percentile(5.0, dist) == 80.0  # 4 of 5 values are <= would need bisect_left semantics
    assert _percentile(100.0, dist) == 100.0


def test_percentile_empty_distribution_returns_nan():
    assert math.isnan(_percentile(1.0, []))


def test_reason_codes_customer_violation_missing_dob():
    row = pd.Series({
        "Alerted Party DOB_missing": True,
        "Hit Details (DOB) (Unresolved)": False,
        "Hit Details (DOB) (MultiValue)": False,
        "Hit Details (Nationality)_missing": False,
        "Matched Screening % (Parsed)": 50.0,
    })
    codes = _reason_codes("CustomerViolation", row)
    assert "ALERTED_PARTY_DOB_MISSING" in codes
    assert "HIGH_SCREENING_MATCH_PERCENTAGE" not in codes


def test_reason_codes_high_screening_match():
    row = pd.Series({
        "Alerted Party DOB_missing": False,
        "Hit Details (DOB) (Unresolved)": False,
        "Hit Details (DOB) (MultiValue)": False,
        "Hit Details (Nationality)_missing": False,
        "Matched Screening % (Parsed)": 99.0,
    })
    codes = _reason_codes("CustomerViolation", row)
    assert codes == ["HIGH_SCREENING_MATCH_PERCENTAGE"]


def test_reason_codes_rule_sheet():
    row = pd.Series({
        "Beneficiary Name_missing": True,
        "Beneficiary Relationship_missing": True,
        "Currency Name_missing": False,
    })
    codes = _reason_codes("Rule", row)
    assert set(codes) == {"BENEFICIARY_NAME_MISSING", "BENEFICIARY_RELATIONSHIP_MISSING"}


def test_plain_language_summary_needs_review_at_or_above_threshold():
    summary = _plain_language_summary(80.0, review_threshold=80.0)
    assert summary["label"] == "Needs Review"


def test_plain_language_summary_not_confident_in_gray_zone():
    # threshold=80, band=15 -> [65, 80) is the gray zone
    summary = _plain_language_summary(70.0, review_threshold=80.0)
    assert summary["label"] == "Not Confident"


def test_plain_language_summary_looks_routine_well_below_threshold():
    summary = _plain_language_summary(10.0, review_threshold=80.0)
    assert summary["label"] == "Looks Routine"


def test_plain_language_summary_never_uses_match_language():
    for pct in (0.0, 30.0, 65.0, 80.0, 100.0):
        summary = _plain_language_summary(pct, review_threshold=80.0)
        combined = (summary["label"] + " " + summary["detail"]).lower()
        assert "match" not in combined


def test_reason_code_plain_text_covers_every_code_reason_codes_can_emit():
    # every code _reason_codes can possibly emit must have a plain-text translation
    row_all_flagged = pd.Series({
        "Alerted Party DOB_missing": True,
        "Hit Details (DOB) (Unresolved)": True,
        "Hit Details (DOB) (MultiValue)": True,
        "Hit Details (Nationality)_missing": True,
        "Matched Screening % (Parsed)": 99.0,
        "Beneficiary Name_missing": True,
        "Beneficiary Relationship_missing": True,
        "Currency Name_missing": True,
    })
    all_codes = _reason_codes("CustomerViolation", row_all_flagged) + _reason_codes("Rule", row_all_flagged)
    for code in all_codes:
        assert code in REASON_CODE_PLAIN_TEXT


def test_score_alert_unknown_alert_type_raises():
    with pytest.raises(ValueError):
        score_alert("not_a_real_type", "x", {})


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scripts/build_scoring_registry.py to have been run locally")
def test_score_alert_missing_required_field_raises():
    with pytest.raises(AlertValidationError) as exc_info:
        score_alert("customer_name", "bad-1", {"UIN": 1})
    assert len(exc_info.value.missing_required_columns) > 0


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scripts/build_scoring_registry.py to have been run locally")
@pytest.mark.parametrize(
    "alert_type,sheet_name",
    [("customer_name", "CustomerViolation"), ("transaction_name", "TransactionNameViolation"), ("transaction_rule", "Rule")],
)
def test_score_alert_real_historical_row_produces_valid_response(alert_type, sheet_name):
    sheets = load_raw_alerts()
    row = sheets[sheet_name].iloc[0]
    fields = allowed_request_fields(alert_type)
    raw = {}
    for c in fields:
        v = row[c]
        if isinstance(v, float) and math.isnan(v):
            v = None
        raw[c] = v

    result = score_alert(alert_type, "test-alert", raw)

    assert result["alert_id"] == "test-alert"
    assert result["alert_type"] == alert_type
    assert 0.0 <= result["novelty"]["global"] <= 100.0
    assert result["recommendation"] in ("REVIEW", "LOWER_TOUCH_CANDIDATE")
    assert result["model_version"]
    assert result["feature_version"]
    assert result["schema_version"]
    assert isinstance(result["reason_codes"], list)
    assert "customer_prior_alert_count" in result["historical_context"]


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scripts/build_scoring_registry.py to have been run locally")
def test_score_alert_reproducible_for_same_input():
    sheets = load_raw_alerts()
    row = sheets["CustomerViolation"].iloc[0]
    fields = allowed_request_fields("customer_name")
    raw = {c: (None if isinstance(row[c], float) and math.isnan(row[c]) else row[c]) for c in fields}

    r1 = score_alert("customer_name", "a", raw)
    r2 = score_alert("customer_name", "a", raw)
    assert r1["novelty"]["global"] == r2["novelty"]["global"]
    assert r1["novelty"]["customer"] == r2["novelty"]["customer"]


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires local scoring registry")
def test_score_alert_evidence_reflects_actual_input_values():
    """Evidence must show the real submitted value, not a derived flag --
    proving a reviewer can trust it as ground truth for what was sent."""
    sheets = load_raw_alerts()
    row = sheets["CustomerViolation"].iloc[10]
    fields = allowed_request_fields("customer_name")
    raw = {c: (None if isinstance(row[c], float) and math.isnan(row[c]) else row[c]) for c in fields}

    result = score_alert("customer_name", "evidence-test", raw)
    assert result["evidence"]["matched_screening_pct"] == raw["Matched Screening %"]


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires local scoring registry")
def test_score_alert_evidence_fields_differ_by_alert_type():
    sheets = load_raw_alerts()

    cv_row = sheets["CustomerViolation"].iloc[10]
    cv_fields = allowed_request_fields("customer_name")
    cv_raw = {c: (None if isinstance(cv_row[c], float) and math.isnan(cv_row[c]) else cv_row[c]) for c in cv_fields}
    cv_result = score_alert("customer_name", "e1", cv_raw)

    rule_row = sheets["Rule"].iloc[3]
    rule_fields = allowed_request_fields("transaction_rule")
    rule_raw = {c: (None if isinstance(rule_row[c], float) and math.isnan(rule_row[c]) else rule_row[c]) for c in rule_fields}
    rule_result = score_alert("transaction_rule", "e2", rule_raw)

    assert "matched_screening_pct" in cv_result["evidence"]
    assert "rule_name" in rule_result["evidence"]
    assert "matched_screening_pct" not in rule_result["evidence"]
    assert "rule_name" not in cv_result["evidence"]
