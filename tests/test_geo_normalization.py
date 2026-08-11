import pandas as pd

from pipelines.normalization.geo import normalize_country_field


def test_code_prefix_pattern_uses_name_half():
    out = normalize_country_field(pd.Series(["IN-INDIA", "AE-UNITED ARAB EMIRATES"]))
    assert out["normalized"].tolist() == ["INDIA", "UNITED ARAB EMIRATES"]
    assert not out["is_multi_value"].any()


def test_bare_known_code_resolved_to_name():
    out = normalize_country_field(pd.Series(["IN", "AE"]))
    assert out["normalized"].tolist() == ["INDIA", "UNITED ARAB EMIRATES"]


def test_bare_full_name_passthrough():
    out = normalize_country_field(pd.Series(["INDIA"]))
    assert out["normalized"].iloc[0] == "INDIA"


def test_multi_value_pipe_and_semicolon_split_deduped():
    out = normalize_country_field(pd.Series(["INDIA | INDIA", "AFGHANISTAN;PAKISTAN"]))
    assert out["normalized"].iloc[0] == "INDIA"
    assert not out["is_multi_value"].iloc[0]  # only one distinct value after dedup
    assert out["is_multi_value"].iloc[1]
    assert out["normalized"].iloc[1] == "AFGHANISTAN | PAKISTAN"


def test_truncated_fragment_flagged_unresolved_not_guessed():
    out = normalize_country_field(
        pd.Series(["AFGHANISTAN | AFGHANISTAN | AFGHANISTAN | AFGHANIS"])
    )
    assert out["unresolved_token_present"].iloc[0]
    # the truncated fragment is preserved, not silently dropped or completed
    assert "AFGHANIS" in out["normalized"].iloc[0]


def test_null_and_blank_pass_through_as_missing():
    out = normalize_country_field(pd.Series([None, "  "]))
    assert pd.isna(out["normalized"].iloc[0])
    assert pd.isna(out["normalized"].iloc[1])
    assert not out["is_multi_value"].iloc[0]
    assert not out["unresolved_token_present"].iloc[0]


def test_unknown_bare_code_left_as_is_not_guessed():
    out = normalize_country_field(pd.Series(["ZZ"]))
    # ZZ is not in the observed-code table -- must not be invented
    assert out["normalized"].iloc[0] == "ZZ"
