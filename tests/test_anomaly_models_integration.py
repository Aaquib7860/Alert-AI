import numpy as np
import pandas as pd

from features.entity_features import build_entity_features
from pipelines.entity.anomaly_models import (
    extract_name_representation_matrix,
    extract_tabular_matrix,
)


def _synthetic_combined_df(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i % 5}" for i in range(n)],
            "Alerted Party Name (Normalized)": [f"NAME {i}" for i in range(n)],
            "Hit Details (Name) (Normalized)": [f"HIT {i}" for i in range(n)],
            "Customer Type": ["INDIVIDUAL"] * n,
            "Alerted Party": ["MemberName"] * n,
            "Alert Type": ["OnBoarding"] * n,
            "Sanctions Screening List Name": ["OFAC"] * n,
            "Branch Name": ["MAIN"] * n,
            "alert_source_sheet": ["CustomerViolation"] * n,
            "Alerted Party Nationality (Normalized)": ["INDIA"] * n,
            "Hit Details (Nationality) (Normalized)": ["INDIA"] * n,
            "Matched Screening % (Parsed)": rng.uniform(50, 100, n),
            "Alerted Party DOB (Parsed)": pd.to_datetime(["1990-01-01"] * n),
            "Hit Details (DOB) (Parsed)": pd.to_datetime([None] * n),
            "Hit Details (DOB) (Year)": pd.array([1990] * n, dtype="Int64"),
            "Hit Details (DOB) (MultiValue)": [False] * n,
            "Hit Details (DOB) (Unresolved)": [False] * n,
            "Alert Generated Date & Time (Parsed)": pd.date_range("2024-01-01", periods=n),
        }
    )


def test_tabular_and_name_blocks_partition_full_matrix_without_overlap():
    df = _synthetic_combined_df()
    matrix, block_names, artifacts = build_entity_features(df, "CombinedEntity")

    tabular = extract_tabular_matrix(matrix, artifacts)
    name = extract_name_representation_matrix(matrix, artifacts)

    assert tabular.shape[0] == matrix.shape[0]
    assert name.shape[0] == matrix.shape[0]
    # together they account for every column exactly once
    assert tabular.shape[1] + name.shape[1] == matrix.shape[1]


def test_tabular_block_excludes_all_name_vocabulary_columns():
    df = _synthetic_combined_df()
    matrix, block_names, artifacts = build_entity_features(df, "CombinedEntity")
    name_width = sum(len(v.vocabulary_) for v in artifacts.name_vectorizers.values())

    tabular = extract_tabular_matrix(matrix, artifacts)
    assert tabular.shape[1] == matrix.shape[1] - name_width
