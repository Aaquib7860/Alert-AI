import numpy as np
import pandas as pd
import pytest

from features.transaction_features import build_transaction_features
from pipelines.transaction.anomaly_models import extract_structured_matrix


def _synthetic_rule_df(n=12):
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "customer_id": [f"c{i % 3}" for i in range(n)],
            "Transaction Type Code": ["ORMTN", "FRXNS"] * (n // 2),
            "Branch Description": ["MAIN"] * n,
            "Currency Name": ["INDIAN RUPEE"] * n,
            "Beneficiary Relationship": ["SELF"] * n,
            "Purpose": ["TRAVEL"] * n,
            "Rule Name": ["53 - AMOUNT", "23 - NATIONALITY"] * (n // 2),
            "Customer Nationality (Normalized)": ["INDIA"] * n,
            "Beneficiary Name (Normalized)": [f"NAME {i}" for i in range(n)],
            "Beneficiary Name_missing": [False] * n,
            "Beneficiary Id Number_missing": [True] * n,
            "Beneficiary Relationship_missing": [False] * n,
            "Currency Name_missing": [False] * n,
            "Purpose_missing": [False] * n,
            "Scan Date (Parsed)": ts,
        }
    )


def test_structured_matrix_excludes_name_and_history_columns():
    df = _synthetic_rule_df()
    matrix, block_names, artifacts = build_transaction_features(df)
    structured = extract_structured_matrix(matrix, artifacts)

    name_width = len(artifacts.beneficiary_name_vectorizer.vocabulary_)
    numeric_block_name = [b for b in block_names if b.startswith("numeric::")][0]
    numeric_width = len(numeric_block_name.split("::", 1)[1].split(","))

    assert structured.shape[1] == matrix.shape[1] - name_width - numeric_width
    assert structured.shape[0] == matrix.shape[0]


def test_structured_matrix_is_prefix_of_full_matrix():
    df = _synthetic_rule_df()
    matrix, block_names, artifacts = build_transaction_features(df)
    structured = extract_structured_matrix(matrix, artifacts)

    # every value in the structured slice must equal the corresponding
    # value in the full matrix's leading columns
    assert (structured != matrix[:, : structured.shape[1]]).nnz == 0


def test_structured_matrix_raises_on_width_mismatch():
    df = _synthetic_rule_df()
    matrix, block_names, artifacts = build_transaction_features(df)
    truncated = matrix[:, :5]  # deliberately too narrow
    with pytest.raises(ValueError):
        extract_structured_matrix(truncated, artifacts)
