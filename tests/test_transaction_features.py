import numpy as np
import pandas as pd
import pytest

from features.transaction_features import (
    BENEFICIARY_NAME_COLUMN,
    CONTEXT_CATEGORICAL_COLUMNS,
    NATIONALITY_COLUMN,
    build_prior_rule_diversity,
    build_transaction_features,
    fit_transaction_feature_artifacts,
    load_transaction_feature_artifacts,
    save_transaction_feature_artifacts,
    transform_transaction_features,
)
from pipelines.validation.temporal import POST_SCORING_COLUMNS


def _synthetic_rule_df():
    ts = pd.to_datetime(
        ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-01", "2024-02-01", "2024-01-15"]
    )
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c2", "c2", "c3"],
            "Transaction Type Code": ["ORMTN", "FRXNS", "ORMTN", "FRXNB", "FRXNB", "ORMTN"],
            "Branch Description": ["MAIN"] * 6,
            "Currency Name": ["INDIAN RUPEE", "US DOLLAR", None, "UAE DIRHAM", "UAE DIRHAM", "US DOLLAR"],
            "Beneficiary Relationship": ["SELF", "FRIEND", "SELF", None, "SISTER", "SELF"],
            "Purpose": ["TRAVEL", "FAMILY SUPPORT", "TRAVEL", "PERSONAL SAVINGS", None, "TRAVEL"],
            "Rule Name": ["53 - AMOUNT", "23 - NATIONALITY", "53 - AMOUNT", "97 - SPLIT", "97 - SPLIT", "53 - AMOUNT"],
            "Customer Nationality (Normalized)": ["INDIA"] * 6,
            "Beneficiary Name (Normalized)": ["JOHN SMITH", "JANE DOE", "JOHN SMITH", None, "SAM LEE", "AMY POE"],
            "Beneficiary Name_missing": [False, False, False, True, False, False],
            "Beneficiary Id Number_missing": [True] * 6,
            "Beneficiary Relationship_missing": [False, False, False, True, False, False],
            "Currency Name_missing": [False, False, True, False, False, False],
            "Purpose_missing": [False, False, False, False, True, False],
            "Scan Date (Parsed)": ts,
        }
    )


def test_no_feature_source_column_is_a_leakage_field():
    leakage = set(POST_SCORING_COLUMNS["Rule"])
    all_cols = CONTEXT_CATEGORICAL_COLUMNS + [NATIONALITY_COLUMN, BENEFICIARY_NAME_COLUMN]
    for col in all_cols:
        assert col.split(" (")[0] not in leakage


def test_comment_field_never_accessed_as_a_column_in_this_module():
    """The `Comment` field is where transaction amount sometimes appears in
    this workbook, but Comment is also a leakage field (post-review text).
    This module must never index into it under any name -- verified by
    scanning the actual source, not just trusting the docstring.
    """
    import inspect

    import features.transaction_features as mod

    source = inspect.getsource(mod)
    assert 'df["Comment"]' not in source
    assert "df['Comment']" not in source
    assert '["Comment"]' not in source


def test_prior_rule_diversity_counts_distinct_rules_only():
    df = _synthetic_rule_df()
    out = build_prior_rule_diversity(df)

    # c1's rule sequence in time order: 53, 23, 53
    # row0 (first c1 alert): 0 distinct rules seen before
    # row1 (second c1 alert): 1 distinct rule seen before (53)
    # row2 (third c1 alert, same rule as row0): 2 distinct rules seen before (53, 23)
    assert out["customer_prior_distinct_rule_count"].loc[0] == 0
    assert out["customer_prior_distinct_rule_count"].loc[1] == 1
    assert out["customer_prior_distinct_rule_count"].loc[2] == 2


def test_prior_rule_diversity_repeated_same_rule_does_not_inflate_count():
    df = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1"],
            "Rule Name": ["53 - AMOUNT", "53 - AMOUNT", "53 - AMOUNT"],
            "Scan Date (Parsed)": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    out = build_prior_rule_diversity(df)
    # same rule every time -> distinct-rule count for the 3rd alert is still 1, not 2
    assert out["customer_prior_distinct_rule_count"].iloc[2] == 1


def test_prior_rule_diversity_nan_when_customer_or_timestamp_missing():
    df = pd.DataFrame(
        {
            "customer_id": [None, "c1"],
            "Rule Name": ["53 - AMOUNT", "53 - AMOUNT"],
            "Scan Date (Parsed)": [pd.Timestamp("2024-01-01"), pd.NaT],
        }
    )
    out = build_prior_rule_diversity(df)
    assert out["customer_prior_distinct_rule_count"].isna().all()


def test_build_transaction_features_shape_and_finiteness():
    df = _synthetic_rule_df()
    matrix, block_names, artifacts = build_transaction_features(df)
    assert matrix.shape[0] == len(df)
    dense = matrix.toarray()
    assert not np.isnan(dense).any()
    assert not np.isinf(dense).any()


def test_unseen_category_at_transform_time_does_not_raise():
    df = _synthetic_rule_df()
    train_df = df.iloc[:3]
    holdout_df = df.iloc[3:].copy()
    holdout_df["Rule Name"] = "999 - NEVER SEEN RULE"

    artifacts = fit_transaction_feature_artifacts(train_df)
    matrix, _ = transform_transaction_features(holdout_df, artifacts)
    assert matrix.shape[0] == len(holdout_df)


def test_save_and_load_round_trip(tmp_path):
    df = _synthetic_rule_df()
    matrix1, block_names1, artifacts = build_transaction_features(df)

    save_transaction_feature_artifacts(artifacts, tmp_path)
    path = tmp_path / f"Rule_{artifacts.feature_version}.joblib"
    assert path.exists()
    manifest_path = tmp_path / f"Rule_{artifacts.feature_version}_manifest.json"
    assert manifest_path.exists()

    loaded = load_transaction_feature_artifacts(path)
    matrix2, block_names2 = transform_transaction_features(df, loaded)
    assert block_names1 == block_names2
    assert (matrix1 != matrix2).nnz == 0
