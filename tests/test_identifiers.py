import pandas as pd
import pytest

from pipelines.ingestion.identifiers import add_stable_ids, find_near_duplicate_candidates


def test_stable_ids_are_namespaced_per_sheet():
    df = pd.DataFrame({"UIN": [1001, 1002], "other": ["a", "b"]})
    out = add_stable_ids(df, "CustomerViolation")
    assert out["customer_id"].iloc[0] == "customerviolation:customer:1001"
    assert out["customer_id"].iloc[1] == "customerviolation:customer:1002"


def test_transaction_id_only_added_when_sheet_has_one():
    df = pd.DataFrame({"UIN": [1], "other": ["a"]})
    out = add_stable_ids(df, "CustomerViolation")
    assert "transaction_id" not in out.columns

    df2 = pd.DataFrame({"UIN": [1], "Trxn Ref Number": [500], "other": ["a"]})
    out2 = add_stable_ids(df2, "TransactionNameViolation")
    assert out2["transaction_id"].iloc[0] == "transactionnameviolation:txn:500"


def test_customer_id_null_when_source_id_missing():
    df = pd.DataFrame(
        {
            "Customer Number": [None, 2000.0],
            "Reference Number": [500, 501],
            "other": ["a", "b"],
        }
    )
    out = add_stable_ids(df, "Rule")
    assert pd.isna(out["customer_id"].iloc[0])
    assert out["customer_id"].iloc[1] == "rule:customer:2000"


def test_record_id_stable_across_row_order():
    df = pd.DataFrame({"UIN": [1, 2], "other": ["a", "b"]})
    out_forward = add_stable_ids(df, "CustomerViolation")
    out_reversed = add_stable_ids(df.iloc[::-1].reset_index(drop=True), "CustomerViolation")

    forward_ids = set(out_forward["record_id"])
    reversed_ids = set(out_reversed["record_id"])
    assert forward_ids == reversed_ids  # same content, same hashes, regardless of order


def test_record_id_differs_for_different_content():
    df = pd.DataFrame({"UIN": [1, 2], "other": ["a", "b"]})
    out = add_stable_ids(df, "CustomerViolation")
    assert out["record_id"].iloc[0] != out["record_id"].iloc[1]


def test_unknown_sheet_raises():
    with pytest.raises(KeyError):
        add_stable_ids(pd.DataFrame({"x": [1]}), "NotRegistered")


def test_near_duplicate_excludes_exact_duplicates():
    df = pd.DataFrame(
        {
            "key": ["A", "A", "B"],
            "other": ["x", "x", "y"],  # rows 0,1 are fully identical
        }
    )
    out = find_near_duplicate_candidates(df, key_columns=["key"])
    # rows 0 and 1 are exact duplicates of each other -- excluded from near-dup flagging
    assert pd.isna(out["near_duplicate_group_id"].iloc[0])
    assert pd.isna(out["near_duplicate_group_id"].iloc[1])


def test_near_duplicate_flags_same_key_different_row():
    df = pd.DataFrame(
        {
            "key": ["A", "A", "B"],
            "other": ["x", "different", "y"],  # rows 0,1 share key but differ elsewhere
        }
    )
    out = find_near_duplicate_candidates(df, key_columns=["key"])
    assert out["near_duplicate_group_id"].iloc[0] == out["near_duplicate_group_id"].iloc[1]
    assert pd.isna(out["near_duplicate_group_id"].iloc[2])  # unique key, no candidate


def test_near_duplicate_ignores_rows_with_missing_key():
    df = pd.DataFrame({"key": [None, None], "other": ["x", "y"]})
    out = find_near_duplicate_candidates(df, key_columns=["key"])
    assert out["near_duplicate_group_id"].isna().all()


def test_near_duplicate_unknown_key_column_raises():
    df = pd.DataFrame({"key": ["A"]})
    with pytest.raises(KeyError):
        find_near_duplicate_candidates(df, key_columns=["not_a_column"])
