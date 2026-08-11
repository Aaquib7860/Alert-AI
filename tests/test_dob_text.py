import datetime as dt

import pandas as pd

from pipelines.normalization.dob_text import classify_hit_dob


def test_clean_datetime_resolved_fully():
    out = classify_hit_dob(pd.Series([dt.datetime(1980, 5, 1)]))
    assert out["parsed_date"].iloc[0] == pd.Timestamp("1980-05-01")
    assert out["year"].iloc[0] == 1980
    assert not out["is_unresolved"].iloc[0]


def test_plausible_bare_year_int_resolved_as_year_only():
    out = classify_hit_dob(pd.Series([1965]))
    assert pd.isna(out["parsed_date"].iloc[0])
    assert out["year"].iloc[0] == 1965
    assert not out["is_unresolved"].iloc[0]


def test_out_of_range_int_flagged_unresolved_not_guessed():
    # 1033879 observed directly in the source data -- not a valid year under
    # any interpretation, must not be silently treated as one
    out = classify_hit_dob(pd.Series([1033879]))
    assert pd.isna(out["year"].iloc[0])
    assert out["is_unresolved"].iloc[0]


def test_multi_value_string_flagged_not_averaged_or_guessed():
    out = classify_hit_dob(pd.Series(["1948 | 1954 | 1960"]))
    assert out["is_multi_value"].iloc[0]
    assert pd.isna(out["year"].iloc[0])
    assert pd.isna(out["parsed_date"].iloc[0])


def test_date_range_flagged_multi_value():
    out = classify_hit_dob(pd.Series(["01 Jan 1963 to 31 Dec 1965"]))
    assert out["is_multi_value"].iloc[0]


def test_circa_qualifier_flagged_multi_value_not_stripped_and_trusted():
    out = classify_hit_dob(pd.Series(["circa 1949"]))
    assert out["is_multi_value"].iloc[0]


def test_malformed_date_flagged_unresolved():
    # day "00" does not exist
    out = classify_hit_dob(pd.Series(["1985/01/00"]))
    assert out["is_unresolved"].iloc[0]
    assert pd.isna(out["parsed_date"].iloc[0])


def test_clean_text_date_parsed_unambiguously():
    out = classify_hit_dob(pd.Series(["02 Dec 1975"]))
    assert out["parsed_date"].iloc[0] == pd.Timestamp("1975-12-02")
    assert out["year"].iloc[0] == 1975


def test_null_passthrough():
    out = classify_hit_dob(pd.Series([None]))
    assert pd.isna(out["parsed_date"].iloc[0])
    assert pd.isna(out["year"].iloc[0])
    assert not out["is_multi_value"].iloc[0]
    assert not out["is_unresolved"].iloc[0]
