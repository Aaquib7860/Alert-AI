import pandas as pd
import pytest

from pipelines.normalization.missingness import add_missingness_indicators


def _customer_violation_base_df():
    # add_missingness_indicators requires every field registered for the
    # sheet to be present -- see module docstring on why this fails loudly
    # rather than silently skipping a missing expected field.
    return pd.DataFrame(
        {
            "Alerted Party DOB": [pd.Timestamp("2020-01-01"), None],
            "Hit Details (DOB)": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
            "Hit Details (Nationality)": ["INDIA", "INDIA"],
        }
    )


def test_missingness_indicator_added_for_registered_field():
    out = add_missingness_indicators(_customer_violation_base_df(), "CustomerViolation")
    assert out["Alerted Party DOB_missing"].tolist() == [False, True]


def test_pob_never_gets_a_missingness_indicator():
    # POB is 100% missing in the current sample -- deliberately excluded
    df = _customer_violation_base_df()
    df["Alerted Party POB"] = [None, None]
    out = add_missingness_indicators(df, "CustomerViolation")
    assert "Alerted Party POB_missing" not in out.columns


def test_unregistered_sheet_returns_unchanged_copy():
    df = pd.DataFrame({"x": [1, 2]})
    out = add_missingness_indicators(df, "NotRegistered")
    assert list(out.columns) == ["x"]


def test_missing_expected_field_raises():
    df = pd.DataFrame({"unrelated": [1]})
    with pytest.raises(KeyError):
        add_missingness_indicators(df, "CustomerViolation")
