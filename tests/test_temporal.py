import pandas as pd
import pytest

from pipelines.validation.temporal import (
    assert_no_post_scoring_columns,
    filter_available_before,
)


def test_leakage_column_in_feature_set_raises():
    with pytest.raises(ValueError):
        assert_no_post_scoring_columns(
            ["Alerted Party Name", "Maker Comment"], "CustomerViolation"
        )


def test_clean_feature_set_passes():
    assert_no_post_scoring_columns(
        ["Alerted Party Name", "Matched Screening %"], "CustomerViolation"
    )  # no raise


def test_rule_sheet_leakage_columns_enforced():
    with pytest.raises(ValueError):
        assert_no_post_scoring_columns(["Comment"], "Rule")
    with pytest.raises(ValueError):
        assert_no_post_scoring_columns(["Action Taken By"], "Rule")


def test_filter_available_before_keeps_only_prior_events():
    df = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
            "cutoff": pd.to_datetime(["2024-06-01"] * 3),
        }
    )
    out = filter_available_before(df, "event_time", "cutoff")
    assert len(out) == 2  # Jan and Jun are <= cutoff, Dec is not


def test_filter_available_before_excludes_null_timestamps():
    df = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2024-01-01"), None],
            "cutoff": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")],
        }
    )
    out = filter_available_before(df, "event_time", "cutoff")
    assert len(out) == 1


def test_filter_available_before_unknown_column_raises():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        filter_available_before(df, "a", "missing_col")
