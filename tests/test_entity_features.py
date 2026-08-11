import numpy as np
import pandas as pd
import pytest

from features.entity_features import (
    CONTEXT_CATEGORICAL_COLUMNS,
    NAME_COLUMNS,
    NATIONALITY_COLUMNS,
    NUMERIC_COLUMNS,
    build_dob_features,
    build_entity_features,
    build_historical_customer_context,
    fit_categorical_encoder,
    fit_entity_feature_artifacts,
    fit_name_vectorizer,
    load_entity_feature_artifacts,
    save_entity_feature_artifacts,
    transform_categorical,
    transform_entity_features,
    transform_names,
)
from pipelines.validation.temporal import POST_SCORING_COLUMNS


def _synthetic_customer_violation_df(n=6):
    ts = pd.to_datetime(
        ["2024-01-01", "2024-02-01", "2024-01-15", "2024-03-01", "2024-01-01", "2024-04-01"]
    )
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c1", "c2", "c3"],
            "Alerted Party Name (Normalized)": ["JOHN SMITH", "JOHN SMITH", "JANE DOE", "JOHN SMITH", "JANE DOE", None],
            "Hit Details (Name) (Normalized)": ["J SMITH", "J SMITH", "J DOE", "J SMITH", "J DOE", "X Y"],
            "Customer Type": ["INDIVIDUAL"] * 6,
            "Alerted Party": ["MemberName"] * 6,
            "Alert Type": ["OnBoarding", "OnGoing", "OnBoarding", "Delta", "OnGoing", "OnBoarding"],
            "Sanctions Screening List Name": ["OFAC"] * 6,
            "Branch Name": ["MAIN"] * 6,
            "Alerted Party Nationality (Normalized)": ["INDIA"] * 6,
            "Hit Details (Nationality) (Normalized)": ["INDIA"] * 6,
            "Matched Screening % (Parsed)": [80.0, 85.0, 90.0, 70.0, 60.0, np.nan],
            "Alerted Party DOB (Parsed)": pd.to_datetime(
                ["1990-01-01", "1990-01-01", "1985-06-01", "1990-01-01", "1985-06-01", None]
            ),
            "Hit Details (DOB) (Parsed)": pd.to_datetime([None] * 6),
            "Hit Details (DOB) (Year)": pd.array([1990, 1990, None, 1990, 1985, None], dtype="Int64"),
            "Hit Details (DOB) (MultiValue)": [False] * 6,
            "Hit Details (DOB) (Unresolved)": [False, False, True, False, False, False],
            "Alert Generated Date & Time (Parsed)": ts,
        }
    )


def test_no_feature_source_column_is_a_leakage_field():
    for sheet_name in NAME_COLUMNS:
        leakage = set(POST_SCORING_COLUMNS.get(sheet_name, []))
        all_cols = (
            NAME_COLUMNS[sheet_name]
            + CONTEXT_CATEGORICAL_COLUMNS[sheet_name]
            + NATIONALITY_COLUMNS[sheet_name]
            + NUMERIC_COLUMNS[sheet_name]
        )
        for col in all_cols:
            assert col.split(" (")[0] not in leakage


def test_dob_features_implausible_age_becomes_nan_not_clipped():
    df = _synthetic_customer_violation_df()
    # inject an implausible DOB (150 years old) -- must not be silently clipped
    df.loc[0, "Alerted Party DOB (Parsed)"] = pd.Timestamp("1870-01-01")
    out = build_dob_features(df, "CustomerViolation")
    assert pd.isna(out["alerted_party_age_years"].iloc[0])


def test_dob_features_missing_dob_flagged():
    df = _synthetic_customer_violation_df()
    out = build_dob_features(df, "CustomerViolation")
    assert out["alerted_party_dob_missing"].iloc[5] == 1
    assert pd.isna(out["alerted_party_age_years"].iloc[5])


def test_dob_features_hit_dob_falls_back_to_year_only():
    df = _synthetic_customer_violation_df()
    out = build_dob_features(df, "CustomerViolation")
    # row 0: Hit Details (DOB) has no parsed date but has year=1990 -> age should resolve
    assert out["hit_dob_resolved"].iloc[0] == 1
    assert not pd.isna(out["hit_age_years"].iloc[0])


def test_historical_context_counts_only_strictly_prior_alerts():
    df = _synthetic_customer_violation_df()
    out = build_historical_customer_context(df, "CustomerViolation")

    # c1 rows are at 2024-01-01 (idx0), 2024-02-01 (idx1), 2024-03-01 (idx3)
    # chronologically: idx0 is first (0 prior), idx1 second (1 prior), idx3 third (2 prior)
    assert out["customer_prior_alert_count"].loc[0] == 0
    assert out["customer_prior_alert_count"].loc[1] == 1
    assert out["customer_prior_alert_count"].loc[3] == 2

    # c3 (idx5) has only itself -- 0 prior
    assert out["customer_prior_alert_count"].loc[5] == 0


def test_historical_context_never_counts_future_alerts_as_prior():
    """The core leakage check for this feature: a customer's LAST
    alert must never show a higher prior-count contribution from an
    alert that happened after it."""
    df = _synthetic_customer_violation_df()
    out = build_historical_customer_context(df, "CustomerViolation")

    c1_rows = df.index[df["customer_id"] == "c1"]
    c1_sorted = df.loc[c1_rows].sort_values("Alert Generated Date & Time (Parsed)")
    counts_in_time_order = out.loc[c1_sorted.index, "customer_prior_alert_count"].tolist()
    assert counts_in_time_order == sorted(counts_in_time_order)  # strictly non-decreasing


def test_historical_context_nan_for_missing_customer_or_timestamp():
    df = pd.DataFrame(
        {
            "customer_id": [None, "c1"],
            "Alert Generated Date & Time (Parsed)": [pd.Timestamp("2024-01-01"), pd.NaT],
        }
    )
    out = build_historical_customer_context(df, "CustomerViolation")
    assert out["customer_prior_alert_count"].isna().all()


def test_name_vectorizer_handles_missing_name_as_zero_vector():
    names = pd.Series(["JOHN SMITH", "JANE DOE", "JOHN SMITH", None])
    vec = fit_name_vectorizer(names)
    matrix = transform_names(vec, names)
    assert matrix.shape[0] == 4
    assert matrix[3].nnz == 0  # missing name -> no fabricated signal


def test_categorical_encoder_unseen_category_does_not_raise():
    train = pd.Series(["A", "B", "A", "B"])
    enc = fit_categorical_encoder(train)
    test_values = pd.Series(["A", "NEVER_SEEN_BEFORE"])
    matrix = transform_categorical(enc, test_values)  # must not raise
    assert matrix.shape[0] == 2
    assert matrix[1].nnz == 0  # unseen category -> all-zero row, not an error


@pytest.mark.parametrize("sheet_name", ["CustomerViolation"])
def test_build_entity_features_shape_matches_row_count(sheet_name):
    df = _synthetic_customer_violation_df()
    matrix, block_names, artifacts = build_entity_features(df, sheet_name)
    assert matrix.shape[0] == len(df)
    assert len(block_names) > 0
    assert not np.isnan(matrix.toarray()).any()
    assert not np.isinf(matrix.toarray()).any()


def test_fit_on_train_transform_on_holdout_no_leakage_of_categories():
    """Simulates a time-forward experiment: fit on an earlier slice, then
    transform a later slice containing a brand-new category. Must not
    raise, and the artifacts must have been fit only on the train slice.
    """
    df = _synthetic_customer_violation_df()
    train_df = df.iloc[:3]
    holdout_df = df.iloc[3:].copy()
    holdout_df["Branch Name"] = "NEVER_SEEN_BRANCH"

    artifacts = fit_entity_feature_artifacts(train_df, "CustomerViolation")
    matrix, block_names = transform_entity_features(holdout_df, "CustomerViolation", artifacts)
    assert matrix.shape[0] == len(holdout_df)


def test_save_and_load_artifacts_round_trip(tmp_path):
    df = _synthetic_customer_violation_df()
    matrix1, block_names1, artifacts = build_entity_features(df, "CustomerViolation")

    save_entity_feature_artifacts(artifacts, tmp_path)
    loaded_path = tmp_path / f"CustomerViolation_{artifacts.feature_version}.joblib"
    assert loaded_path.exists()

    manifest_path = tmp_path / f"CustomerViolation_{artifacts.feature_version}_manifest.json"
    assert manifest_path.exists()

    loaded_artifacts = load_entity_feature_artifacts(loaded_path)
    matrix2, block_names2 = transform_entity_features(df, "CustomerViolation", loaded_artifacts)

    assert block_names1 == block_names2
    assert (matrix1 != matrix2).nnz == 0  # identical transform after reload
