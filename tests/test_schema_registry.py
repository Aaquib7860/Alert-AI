import pandas as pd
import pytest

from pipelines.validation.schema_registry import (
    SCHEMA_REGISTRY,
    validate_all,
    validate_schema,
)


def _minimal_valid_customer_violation_df() -> pd.DataFrame:
    schema = SCHEMA_REGISTRY["CustomerViolation"]
    return pd.DataFrame({c.name: [None] for c in schema.columns})


def test_valid_schema_passes():
    df = _minimal_valid_customer_violation_df()
    result = validate_schema(df, "CustomerViolation")
    assert result.passed
    assert result.missing_required_columns == []


def test_missing_required_column_fails():
    df = _minimal_valid_customer_violation_df().drop(columns=["Alerted Party Name"])
    result = validate_schema(df, "CustomerViolation")
    assert not result.passed
    assert "Alerted Party Name" in result.missing_required_columns


def test_missing_optional_column_still_passes():
    df = _minimal_valid_customer_violation_df().drop(columns=["Alerted Party POB"])
    result = validate_schema(df, "CustomerViolation")
    assert result.passed


def test_unexpected_column_reported_but_does_not_fail():
    df = _minimal_valid_customer_violation_df()
    df["Some New Client Field"] = None
    result = validate_schema(df, "CustomerViolation")
    assert result.passed
    assert "Some New Client Field" in result.unexpected_columns


def test_unknown_sheet_name_raises():
    with pytest.raises(KeyError):
        validate_schema(pd.DataFrame(), "NotARealSheet")


def test_validate_all_covers_every_registered_sheet():
    sheets = {
        name: _valid_df_for(name) for name in SCHEMA_REGISTRY
    }
    results = validate_all(sheets)
    assert set(results.keys()) == set(SCHEMA_REGISTRY.keys())
    assert all(r.passed for r in results.values())


def _valid_df_for(sheet_name: str) -> pd.DataFrame:
    schema = SCHEMA_REGISTRY[sheet_name]
    return pd.DataFrame({c.name: [None] for c in schema.columns})


def test_leakage_columns_identified_per_sheet():
    cv = SCHEMA_REGISTRY["CustomerViolation"]
    assert "Maker Comment" in cv.leakage_columns
    assert "Alert Closure Date & Time" in cv.leakage_columns
    assert "Alerted Party Name" not in cv.leakage_columns

    rule = SCHEMA_REGISTRY["Rule"]
    # Status (Released/UPS/Followup) is a post-review outcome, not a safe
    # live feature -- see Phase 1 label audit -- so it is leakage too.
    assert set(rule.leakage_columns) == {
        "Comment", "Actiondate", "Action Taken By", "Status",
    }
